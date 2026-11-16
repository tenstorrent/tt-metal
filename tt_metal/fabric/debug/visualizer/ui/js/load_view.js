// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The load page's capture list, and the progress line while a capture loads.

import { element, formatBytes, formatCount, formatUtcTime, parseCompactUtcTime } from "./format.js";

// Draws one button per run of each capture in `listing`, under a heading and the folder searched; onOpen(capture, run)
// is called when one is clicked.
export function renderCaptureList(container, listing, heading, onOpen) {
    const options = listing.captures.flatMap((capture) => capture.runs.map((run) => describeRunOption(capture, run, onOpen)));
    const children = [
        element("h2", { className: "label" }, heading),
        element("p", { className: "searched-folder" }, listing.root),
        options.length > 0
            ? element("div", { className: "capture-list" }, options)
            : element("p", { className: "empty-list" },
                "No decoded files here. A capture's decoded files are named fabric_decoded_rank_<r>_of_<n>.json."),
        listing.search_stopped_early
            ? element("p", { className: "note" },
                "The search stopped before it reached every folder, so some captures may be missing. Pass a --files-path closer to the captures.")
            : null,
    ];
    container.replaceChildren(...children.filter(Boolean));
}

function describeRunOption(capture, run, onOpen) {
    const worldSize = run.ranks[0].world_size;
    const foundRankNumbers = new Set(run.ranks.map((rankFile) => rankFile.rank));
    const missingRankNumbers = [];
    for (let rankNumber = 1; rankNumber <= worldSize; rankNumber++) {
        if (!foundRankNumbers.has(rankNumber)) missingRankNumbers.push(rankNumber);
    }
    const decodedBytes = run.ranks.reduce((total, rankFile) => total + rankFile.decoded.size_bytes, 0);
    const ranksWithoutBin = run.ranks.filter((rankFile) => !rankFile.bin).map((rankFile) => rankFile.rank);

    const facts = [
        run.run_id ? `run ${run.run_id}` : "no run folder",
        `${run.ranks.length} of ${formatCount(worldSize, "rank")}` +
            (missingRankNumbers.length ? `, missing ${missingRankNumbers.join(", ")}` : ""),
        `${formatBytes(decodedBytes)} decoded`,
        ranksWithoutBin.length === 0
            ? "every .bin found"
            : `no .bin for rank ${ranksWithoutBin.join(", ")}`,
    ];
    const button = element("button", { type: "button", className: "capture-option" },
        element("span", { className: "capture-time" }, formatUtcTime(parseCompactUtcTime(capture.capture_id))),
        element("span", { className: "capture-id" }, capture.capture_id),
        element("span", { className: "capture-facts" }, facts.join(" · ")),
        element("span", { className: "capture-folder" }, run.folder),
    );
    button.addEventListener("click", () => onOpen(capture, run));
    return button;
}

// Shows how far loading has got: the rank being read or parsed, and the bytes read out of every rank's total.
export function renderLoadProgress(statusElement, { rankFile, step, bytesDone, totalBytes }) {
    const fraction = totalBytes > 0 ? Math.min(bytesDone / totalBytes, 1) : 0;
    const stepText = step === "parsing"
        ? `Parsing rank ${rankFile.rank} of ${rankFile.world_size}`
        : `Reading rank ${rankFile.rank} of ${rankFile.world_size}`;
    statusElement.className = "load-status";
    statusElement.replaceChildren(
        element("span", {}, `${stepText}: ${formatBytes(bytesDone)} of ${formatBytes(totalBytes)}`),
        element("span", { className: "progress" }, element("span", { style: { width: `${(fraction * 100).toFixed(1)}%` } })),
    );
}

export function renderLoadError(statusElement, message) {
    statusElement.className = "load-status error";
    statusElement.replaceChildren(message);
}

export function clearLoadStatus(statusElement) {
    statusElement.className = "load-status";
    statusElement.replaceChildren();
}
