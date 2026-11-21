// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One display for every field a router, channel or edge has, whatever its kind (l1, stream_reg, number, flag, enum).
// Enums show by name, l1 structs as a nested list, and a field the UI doesn't know about still shows what it holds.
// A field is null when the builder didn't allocate what it describes.

import { element, formatHex } from "./format.js";

const CATEGORY_NAMES = {
    lifecycle: "Lifecycle",
    kernel_params: "Kernel parameters",
    flow_control: "Flow control",
    control_info: "Control info",
    diagnostics: "Diagnostics",
    not_allocated: "Not allocated",
};

// A table of `fieldsByErisc` ({ erisc0: { name: field }, ... }): one row per field name, one column per ERISC.
export function renderFieldTable(fieldsByErisc, fieldNames = null) {
    const eriscs = Object.keys(fieldsByErisc ?? {});
    const names = fieldNames ?? [...new Set(eriscs.flatMap((erisc) => Object.keys(fieldsByErisc[erisc] ?? {})))];
    return element("table", { className: "field-table" },
        eriscs.length > 1 ? element("thead", {}, element("tr", {}, element("th", { scope: "col" }, "Field"),
            eriscs.map((erisc) => element("th", { scope: "col" }, erisc)))) : null,
        element("tbody", {}, names.map((name) => element("tr", {},
            element("th", { scope: "row" }, breakAfterUnderscores(name)),
            eriscs.map((erisc) => {
                const fields = fieldsByErisc[erisc] ?? {};
                return element("td", {}, name in fields ? renderFieldValue(fields[name]) : element("span", { className: "faint" }, "—"));
            })))));
}

// A field name that may wrap only after an underscore: "producer_credit_" then "return_noc", never mid-word.
function breakAfterUnderscores(name) {
    return name.split(/(?<=_)/).flatMap((part, index) => (index ? [document.createElement("wbr"), part] : [part]));
}

// The router's fields, one collapsed section per category in the vocabulary's order, each a renderFieldTable.
export function renderFieldsByCategory(fieldsByErisc, categoryOrder) {
    const namesByCategory = new Map();
    for (const fields of Object.values(fieldsByErisc ?? {})) {
        for (const [name, field] of Object.entries(fields ?? {})) {
            // A null field has no category: the builder didn't allocate what it describes.
            const category = field === null ? "not_allocated" : field?.category ?? "other";
            if (!namesByCategory.has(category)) namesByCategory.set(category, new Set());
            namesByCategory.get(category).add(name);
        }
    }
    const categories = [...(categoryOrder ?? []), ...[...namesByCategory.keys()].filter((name) => !categoryOrder?.includes(name))]
        .filter((category) => namesByCategory.has(category));
    return categories.map((category) => {
        const names = [...namesByCategory.get(category)];
        return element("details", { className: "field-group" },
            element("summary", {}, CATEGORY_NAMES[category] ?? category, element("span", { className: "faint" }, ` · ${names.length}`)),
            renderFieldTable(fieldsByErisc, names));
    });
}

// What one field holds, as a node.
export function renderFieldValue(field) {
    if (field === null || field === undefined) return element("span", { className: "faint", title: "The builder didn't allocate it" }, "not allocated");
    const where = describeWhere(field);
    switch (field.kind) {
        case "flag":
            return element("span", {}, field.value ? "yes" : "no");
        case "number":
            return element("span", {}, String(field.value));
        case "enum":
            return element("span", {}, field.value_name ?? String(field.value), field.value_name ? element("span", { className: "faint" }, ` (${field.value})`) : null);
        case "stream_reg":
            return element("span", {}, describeStreamRegister(field.decoded), where);
        case "l1":
            return element("span", { className: "l1-value" }, renderL1Decoded(field.decoded), where);
        default:
            // A kind the UI doesn't know: whatever value or decoded value it has.
            return element("span", {}, renderPlainValue(field.value ?? field.decoded ?? field));
    }
}

function describeWhere(field) {
    const parts = [];
    if (field.stream_id !== undefined) parts.push(`stream ${field.stream_id}`);
    if (field.address !== undefined && field.kind === "l1") parts.push(`at ${formatHex(field.address)}`);
    if (field.active === false || field.decoded?.active === false) parts.push("inactive");
    return parts.length ? element("span", { className: "faint where" }, ` · ${parts.join(" · ")}`) : null;
}

// A count register shows its count of depth and what decode derived from it; any other shows its value.
function describeStreamRegister(decoded) {
    if (!decoded) return "not read";
    if ("count" in decoded) {
        const parts = [`${decoded.count}${decoded.depth !== null && decoded.depth !== undefined ? ` of ${decoded.depth}` : ""}`];
        for (const name of ["unsent", "uncredited", "taken_at", "signed"]) {
            if (decoded[name] !== undefined && decoded[name] !== null) parts.push(`${name.replace("_", " ")} ${decoded[name]}`);
        }
        return parts.join(" · ");
    }
    if (decoded.value !== undefined) return renderPlainValue(decoded.value);
    if (decoded.phase !== undefined || decoded.step !== undefined) return `phase ${decoded.phase ?? "none"}, step ${decoded.step ?? "none"}`;
    return `raw ${decoded.raw}`;
}

function renderL1Decoded(decoded) {
    if (!decoded) return "not read";
    if (decoded.value !== undefined) return renderPlainValue(decoded.value);
    if (decoded.elements !== undefined) return renderPlainValue(decoded.elements);
    if (decoded.size_bytes !== undefined) return `${decoded.size_bytes} bytes, ${decoded.nonzero_bytes ?? "?"} not zero`;
    if (decoded.rounds !== undefined) return `${decoded.rounds.length} rounds`;
    return renderPlainValue(decoded);
}

// A value from the decoded file: an enumerator as its name, a NoC coordinate as (x, y), a struct as a nested list, a
// short list inline. A value whose key names an address shows in hex.
export function renderPlainValue(value, key = "") {
    if (value === null || value === undefined) return element("span", { className: "faint" }, "null");
    if (typeof value === "number" && /address|addr$/.test(key)) return formatHex(value);
    if (typeof value !== "object") return String(value);
    if (!Array.isArray(value) && Object.keys(value).length === 2 && "x" in value && "y" in value) return `(${value.x}, ${value.y})`;
    if (!Array.isArray(value) && "name" in value && "value" in value && Object.keys(value).length === 2) {
        return element("span", {}, String(value.name), element("span", { className: "faint" }, ` (${value.value})`));
    }
    if (Array.isArray(value)) {
        if (value.every((item) => item === null || typeof item !== "object") && value.length <= 16) return `[${value.join(", ")}]`;
        return element("details", { className: "nested" }, element("summary", {}, `${value.length} elements`),
            element("ol", { start: 0 }, value.map((item) => element("li", {}, renderPlainValue(item)))));
    }
    return element("dl", { className: "nested-struct" }, Object.entries(value).map(([name, item]) =>
        element("div", {}, element("dt", {}, name), element("dd", {}, renderPlainValue(item, name)))));
}
