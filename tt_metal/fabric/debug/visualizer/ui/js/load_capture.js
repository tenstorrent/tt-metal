// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reads and checks every rank's decoded file of one run of a capture. A problem that makes the capture unreadable
// throws a CaptureLoadError; one that only limits what can be shown is returned as a warning.

import { formatCount } from "./format.js";

export const READABLE_DECODED_VERSION = 1;

export class CaptureLoadError extends Error {}

// Returns { captureListing, run, ranks, missingRankNumbers, warnings }. Each rank is { number, rankFile, decoded, bin },
// where bin is { readable, reason }: raw bytes are read from the rank's .bin only when it is readable.
// reportProgress({ rankFile, step, bytesDone, totalBytes }) is called while the files are read and parsed.
export async function loadCapture(source, captureListing, run, reportProgress) {
    const worldSizes = new Set(run.ranks.map((rankFile) => rankFile.world_size));
    if (worldSizes.size !== 1) {
        throw new CaptureLoadError(
            `The folder holds rank files of different world sizes (${[...worldSizes].join(", ")}), from more than one run.`);
    }
    const [worldSize] = worldSizes;
    const foundRankNumbers = new Set(run.ranks.map((rankFile) => rankFile.rank));
    const missingRankNumbers = [];
    for (let rankNumber = 1; rankNumber <= worldSize; rankNumber++) {
        if (!foundRankNumbers.has(rankNumber)) missingRankNumbers.push(rankNumber);
    }

    const totalBytes = run.ranks.reduce((total, rankFile) => total + rankFile.decoded.size_bytes, 0);
    let bytesBeforeThisRank = 0;
    const ranks = [];
    for (const rankFile of run.ranks) {
        const report = (step, bytesDone) => reportProgress?.({ rankFile, step, bytesDone, totalBytes });
        report("reading", bytesBeforeThisRank);
        let text;
        try {
            text = await source.readDecodedText(rankFile, (bytesRead) => report("reading", bytesBeforeThisRank + bytesRead));
        } catch (error) {
            throw new CaptureLoadError(`Couldn't read ${rankFile.decoded.name}: ${error.message}`);
        }
        report("parsing", bytesBeforeThisRank + rankFile.decoded.size_bytes);
        await waitForPaint();
        let decoded;
        try {
            decoded = JSON.parse(text);
        } catch (error) {
            throw new CaptureLoadError(`${rankFile.decoded.name} isn't valid JSON: ${error.message}`);
        }
        checkDecodedFile(rankFile, decoded);
        ranks.push({ number: rankFile.rank, rankFile, decoded, bin: checkBin(rankFile, decoded) });
        bytesBeforeThisRank += rankFile.decoded.size_bytes;
    }

    const captureIds = new Set(ranks.map((rank) => rank.decoded.capture?.capture_id));
    if (captureIds.size !== 1) {
        throw new CaptureLoadError(
            `The ranks come from different captures (${[...captureIds].join(", ")}). Each rank of a run must be from the same capture.`);
    }

    const warnings = [];
    if (missingRankNumbers.length > 0) {
        warnings.push(
            missingRankNumbers.length === 1
                ? `Rank ${missingRankNumbers[0]} of ${worldSize} is missing, so the chips it owns show without routers.`
                : `Ranks ${missingRankNumbers.join(", ")} of ${worldSize} are missing, so the chips they own show without routers.`);
    }
    for (const rank of ranks) {
        if (!rank.bin.readable) warnings.push(`Rank ${rank.number}: ${rank.bin.reason}, so its raw bytes can't be shown.`);
    }
    return { captureListing, run, ranks, missingRankNumbers, warnings };
}

function checkDecodedFile(rankFile, decoded) {
    const name = rankFile.decoded.name;
    if (decoded?.file_kind !== "fabric_decoded") {
        throw new CaptureLoadError(`${name} isn't a decoded file: its file_kind is ${JSON.stringify(decoded?.file_kind)}.`);
    }
    if (decoded.decoded_version !== READABLE_DECODED_VERSION) {
        throw new CaptureLoadError(
            `${name} has decoded_version ${decoded.decoded_version}, and this UI reads version ${READABLE_DECODED_VERSION}.`);
    }
    // File names number ranks from 1; mpi_rank numbers them from 0.
    if (decoded.run?.world_size !== rankFile.world_size || decoded.run?.mpi_rank !== rankFile.rank - 1) {
        throw new CaptureLoadError(
            `${name} says it is mpi_rank ${decoded.run?.mpi_rank} of world size ${decoded.run?.world_size}, ` +
            "which doesn't match its name.");
    }
}

function checkBin(rankFile, decoded) {
    const expected = decoded.inputs?.bin;
    if (!expected) return { readable: false, reason: "the decoded file doesn't name its .bin" };
    if (!rankFile.bin || rankFile.bin.name !== expected.file) {
        return { readable: false, reason: `${expected.file} isn't beside the decoded file` };
    }
    if (rankFile.bin.size_bytes !== expected.size_bytes) {
        return {
            readable: false,
            reason: `${expected.file} is ${formatCount(rankFile.bin.size_bytes, "byte")}, and decode read ${formatCount(expected.size_bytes, "byte")}`,
        };
    }
    return { readable: true, reason: null };
}

// Lets the browser show the latest progress before a long synchronous step such as JSON.parse. A tab in the background
// gets no animation frames, so the timeout lets loading go on there.
function waitForPaint() {
    return new Promise((resolve) => {
        requestAnimationFrame(() => setTimeout(resolve));
        setTimeout(resolve, 100);
    });
}
