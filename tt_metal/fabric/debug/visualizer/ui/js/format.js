// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Formatting for numbers, sizes and times, and a helper that builds DOM elements without innerHTML.

export function formatBytes(sizeBytes) {
    if (sizeBytes >= 1e6) return `${(sizeBytes / 1e6).toFixed(1)} MB`;
    if (sizeBytes >= 1e3) return `${(sizeBytes / 1e3).toFixed(1)} KB`;
    return `${sizeBytes} B`;
}

export function formatHex(value) {
    return typeof value === "number" ? `0x${value.toString(16)}` : String(value);
}

export function formatCount(count, singular, plural = `${singular}s`) {
    return `${count.toLocaleString("en-US")} ${count === 1 ? singular : plural}`;
}

// Capture ids and run ids start with a UTC time written as 20261113T230100Z.
export function parseCompactUtcTime(text) {
    const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/.exec(text ?? "");
    if (!match) return null;
    const [, year, month, day, hour, minute, second] = match.map(Number);
    return new Date(Date.UTC(year, month - 1, day, hour, minute, second));
}

export function formatUtcTime(date) {
    if (!date || Number.isNaN(date.getTime())) return "unknown time";
    return `${date.toISOString().slice(0, 19).replace("T", " ")} UTC`;
}

export function formatDuration(milliseconds) {
    if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
    const seconds = milliseconds / 1000;
    if (seconds < 120) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
    return `${Math.round(seconds / 60)} min`;
}

// Escapes text for SVG or HTML markup built as a string.
export function escapeXml(text) {
    return String(text).replace(/[&<>"']/g, (character) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}

// element("div", { className: "fact" }, "text", childElement): properties are assigned to the element, except
// `dataset` and `style`, which are merged into it. Strings become text nodes, so file contents are never parsed as HTML.
export function element(tagName, properties = {}, ...children) {
    const created = document.createElement(tagName);
    for (const [name, value] of Object.entries(properties)) {
        if (name === "dataset" || name === "style") Object.assign(created[name], value);
        else created[name] = value;
    }
    created.append(...children.flat().filter((child) => child !== null && child !== undefined && child !== false));
    return created;
}
