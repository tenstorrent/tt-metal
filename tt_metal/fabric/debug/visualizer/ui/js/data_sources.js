// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The UI reads captures through one interface, whether serve.py lists them or the viewer picked or dropped a folder:
//
//   listCaptures()                         { root, search_stopped_early, captures: [{ capture_id, folder, runs }] },
//                                          each run { run_id, folder, ranks }, the shape serve.py's /api/captures has
//   readDecodedText(rankFile, onProgress)  the rank's decoded file as text; onProgress(bytesRead) while it reads
//   readBinBytes(rankFile, offset, size)   an ArrayBuffer holding those bytes of the rank's .bin
//
// A rank file is { rank, world_size, decoded, bin }. `decoded` and `bin` each have a name and size_bytes, and either a
// URL (served) or a File (picked); `bin` is null when there is no .bin beside the decoded file.

const DECODED_FILE_PATTERN = /^fabric_decoded_rank_(\d+)_of_(\d+)\.json$/;
const CAPTURE_FILE_PATTERN = /^fabric_(?:decoded_rank_\d+_of_\d+\.json|capture_rank_\d+_of_\d+\.bin)$/;

export function createServedSource() {
    return {
        kind: "served",

        async listCaptures() {
            const response = await fetch("api/captures", { cache: "no-store" });
            if (!response.ok) throw new Error(`the server answered ${response.status} ${response.statusText}`);
            return response.json();
        },

        async readDecodedText(rankFile, onProgress) {
            const response = await fetch(rankFile.decoded.url, { cache: "no-store" });
            if (!response.ok) {
                throw new Error(`couldn't read ${rankFile.decoded.name}: ${response.status} ${response.statusText}`);
            }
            // The body arrives decompressed, so the bytes read count toward the decoded file's size on disk.
            const reader = response.body.getReader();
            const chunks = [];
            let bytesRead = 0;
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                bytesRead += value.length;
                onProgress?.(bytesRead);
            }
            return new Blob(chunks).text();
        },

        async readBinBytes(rankFile, offset, size) {
            const response = await fetch(rankFile.bin.url, {
                headers: { Range: `bytes=${offset}-${offset + size - 1}` },
            });
            if (response.status !== 206) {
                throw new Error(`couldn't read ${size} bytes at ${offset} of ${rankFile.bin.name}: ${response.status}`);
            }
            return response.arrayBuffer();
        },
    };
}

// `pickedFiles` is [{ path, file }], with paths relative to the parent of the folder the viewer picked or dropped.
export function createFolderSource(folderName, pickedFiles) {
    const listing = listPickedCaptures(folderName, pickedFiles);
    return {
        kind: "folder",

        async listCaptures() {
            return listing;
        },

        async readDecodedText(rankFile, onProgress) {
            const text = await rankFile.decoded.file.text();
            onProgress?.(rankFile.decoded.size_bytes);
            return text;
        },

        async readBinBytes(rankFile, offset, size) {
            return rankFile.bin.file.slice(offset, offset + size).arrayBuffer();
        },
    };
}

// Keeps the files a capture is made of, from an <input webkitdirectory> file list.
export function collectPickedFiles(fileList) {
    return [...fileList]
        .filter((file) => CAPTURE_FILE_PATTERN.test(file.name) && !isInHiddenFolder(file.webkitRelativePath))
        .map((file) => ({ path: file.webkitRelativePath || file.name, file }));
}

// Walks the folders dropped on the page and keeps the files a capture is made of. Reads the drop's entries before its
// first await: the browser empties the DataTransfer once the drop handler returns.
export async function collectDroppedFiles(dataTransfer) {
    const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
    const pickedFiles = [];
    for (const entry of entries) await collectEntryFiles(entry, pickedFiles);
    const folderName = entries.length === 1 ? entries[0].name : `${entries.length} dropped items`;
    return { folderName, pickedFiles };
}

async function collectEntryFiles(entry, pickedFiles) {
    if (entry.name.startsWith(".")) return;
    if (entry.isFile) {
        if (!CAPTURE_FILE_PATTERN.test(entry.name)) return;
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        pickedFiles.push({ path: entry.fullPath.replace(/^\//, ""), file });
        return;
    }
    const reader = entry.createReader();
    // readEntries returns a folder's entries in batches, and an empty batch at the end.
    for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (batch.length === 0) break;
        for (const child of batch) await collectEntryFiles(child, pickedFiles);
    }
}

function isInHiddenFolder(relativePath) {
    return relativePath.split("/").some((part) => part.startsWith("."));
}

// Groups picked decoded files the way serve.py does: a capture is one capture id under one runs folder, in the run
// layout runs/<run_id>/captures/<capture_id>/ or the copy layout <capture_id>/<run_id>/captures/<capture_id>/. A
// picked capture folder on its own has no run folder above it, so its run id is null.
function listPickedCaptures(folderName, pickedFiles) {
    const filesByPath = new Map(pickedFiles.map(({ path, file }) => [path, file]));
    const captures = new Map();
    for (const { path, file } of pickedFiles) {
        const parts = path.split("/");
        const match = DECODED_FILE_PATTERN.exec(parts.at(-1));
        if (!match) continue;
        const captureFolderParts = parts.slice(0, -1);
        const captureId = captureFolderParts.at(-1) ?? folderName;
        const runFolderParts = captureFolderParts.at(-2) === "captures" ? captureFolderParts.slice(0, -2) : null;
        const runsFolder = (runFolderParts ?? captureFolderParts).slice(0, -1).join("/");
        const runId = runFolderParts?.at(-1) ?? null;

        const captureKey = `${runsFolder}\n${captureId}`;
        if (!captures.has(captureKey)) {
            captures.set(captureKey, { capture_id: captureId, folder: runsFolder || folderName, runs: new Map() });
        }
        const runs = captures.get(captureKey).runs;
        if (!runs.has(runId)) {
            runs.set(runId, { run_id: runId, folder: (runFolderParts ?? captureFolderParts).join("/"), ranks: [] });
        }

        const [, rankNumber, worldSize] = match;
        const binName = `fabric_capture_rank_${rankNumber}_of_${worldSize}.bin`;
        const binFile = filesByPath.get([...captureFolderParts, binName].join("/"));
        runs.get(runId).ranks.push({
            rank: Number(rankNumber),
            world_size: Number(worldSize),
            decoded: { name: file.name, size_bytes: file.size, file },
            bin: binFile ? { name: binName, size_bytes: binFile.size, file: binFile } : null,
        });
    }

    const listing = [...captures.values()].map((capture) => ({
        ...capture,
        runs: [...capture.runs.values()]
            .sort((first, second) => String(first.run_id).localeCompare(String(second.run_id)))
            .map((run) => ({ ...run, ranks: run.ranks.sort((first, second) => first.rank - second.rank) })),
    }));
    // Capture ids are UTC times, so the newest sorts first when reversed, as serve.py lists them.
    listing.sort((first, second) =>
        second.capture_id.localeCompare(first.capture_id) || second.folder.localeCompare(first.folder));
    return { root: folderName, search_stopped_early: false, captures: listing };
}
