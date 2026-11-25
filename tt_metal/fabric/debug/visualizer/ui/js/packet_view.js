// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A ring buffer slot's packet header, as decode wrote it: the header's own members, its NoC command, its route, and
// the path the route takes, hop by hop, from the receiver that reads the packet next. The path is the route as
// written; beside each hop the view shows what decode leaves to it, the hop router's live directions and whether its
// receiver has forwarding disabled. A slot's header is the last one written to it, whether or not its packet is
// still there.

import { element, formatHex } from "./format.js";
import { renderPlainValue } from "./field_view.js";

const PATH_END_NOTES = {
    delivered: "the hop delivers to its chip and sends the packet nowhere else",
    no_action: "the hop's action is empty: nothing to deliver and nowhere to forward",
    off_mesh: "the route leaves the edge of a mesh that doesn't wrap that way",
    unknown_chip: "the route crosses a Z link decode can't follow",
    loops: "the route reaches a chip and side it already reached",
    leaves_mesh: "the route crosses an intermesh link, whose far side rewrites the route",
};
const PLANAR_DIRECTIONS = ["east", "west", "north", "south", "z"];

// The router path without its mesh, "C1/E0", for a table that has many.
function shortRouterPath(path) {
    return path ? path.split("/").slice(1).join("/") : "";
}

// What one slot's header says, and its path. `slot` is { router, kind ("sender" | "receiver"), index, address,
// header }. onShowPath() draws the path on the hardware view; onSelectRouter(path) opens a hop's router.
export function renderSlotHeader(model, slot, { onSelectRouter, onShowPath }) {
    const { router, kind, header } = slot;
    if (!header) {
        return element("div", { className: "slot-detail" },
            element("p", { className: "hint" }, "Never written: the slot's header is all zeros, as routers leave it at startup."));
    }
    const showButton = element("button", { type: "button", className: "ghost" }, "Show on hardware");
    showButton.addEventListener("click", onShowPath);
    const firstHop = header.path?.[0];
    const startNote = kind === "sender"
        ? `The path starts at the receiver that reads the packet next, ${firstHop?.router ?? firstHop?.chip ?? "the peer"}, across ${router.name}'s link.`
        : `The path starts here, at ${router.name}'s receiver.`;
    return element("div", { className: "slot-detail" },
        element("div", { className: "slot-detail-head" },
            element("b", {}, `Slot ${slot.index}`), element("span", { className: "faint" }, ` · ${formatHex(slot.address)}`), showButton),
        element("p", { className: "hint" }, "The header last written to this slot. The capture doesn't say whether its packet is still here."),
        element("div", { className: "header-grid" },
            element("div", {}, element("p", { className: "column-title" }, "Header"), renderHeaderMembers(header)),
            element("div", {}, element("p", { className: "column-title" }, "Command"), renderCommand(header.command)),
            element("div", {}, element("p", { className: "column-title" }, "Route"), renderRoute(header.route))),
        element("p", { className: "column-title" }, `Path · ${describePathShape(header.path)}`),
        element("p", { className: "hint" }, startNote),
        renderPathList(model, header.path ?? [], onSelectRouter));
}

function keyValues(entries) {
    return element("dl", { className: "key-values" }, entries.filter(([, value]) => value !== undefined)
        .map(([key, value]) => element("div", {}, element("dt", {}, key), element("dd", {}, value))));
}

function renderHeaderMembers(header) {
    const known = new Set(["command", "route", "path"]);
    return keyValues(Object.entries(header).filter(([key]) => !known.has(key)).map(([key, value]) => [key,
        key === "payload_size_bytes" && typeof value === "number" ? `${value.toLocaleString("en-US")} B` : renderPlainValue(value, key)]));
}

// A NoC address { x, y, address } reads as "(x, y) 0x…"; anything else as the decoded file has it.
function isNocAddress(value) {
    return value && typeof value === "object" && !Array.isArray(value) && "x" in value && "y" in value && "address" in value
        && Object.keys(value).length === 3;
}

function renderCommandValue(value, key) {
    if (isNocAddress(value)) return `(${value.x}, ${value.y}) ${formatHex(value.address)}`;
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (Array.isArray(value)) {
        return element("ol", { className: "command-list", start: 0 }, value.map((item) => element("li", {}, renderCommandValue(item, key))));
    }
    if (value && typeof value === "object") {
        return keyValues(Object.entries(value).map(([name, item]) => [name, renderCommandValue(item, name)]));
    }
    return renderPlainValue(value, key);
}

function renderCommand(command) {
    if (!command) return element("span", { className: "faint" }, "none");
    return keyValues(Object.entries(command).map(([key, value]) => [key, renderCommandValue(value, key)]));
}

// 1D: one action per hop. 2D: the destination, multicast hop counts, and the two maps' non-empty entries: `y` by row,
// `x` by column.
function renderRoute(route) {
    if (!route) return element("span", { className: "faint" }, "none");
    if (Array.isArray(route.actions)) return keyValues([["actions", route.actions.join(" → ") || "none"]]);
    const entries = [];
    if ("dst" in route) entries.push(["dst", renderPlainValue(route.dst)]);
    if ("mcast" in route) {
        const counts = Object.entries(route.mcast ?? {}).filter(([, hops]) => hops);
        entries.push(["mcast", counts.length ? counts.map(([direction, hops]) => `${direction} ${hops}`).join(", ") : "none"]);
    }
    for (const [map, axis] of [["y", "row"], ["x", "column"]]) {
        if (!Array.isArray(route[map])) continue;
        const written = route[map].map((actions, index) => [index, actions]).filter(([, actions]) => actions?.length);
        entries.push([map, written.length
            ? element("span", { className: "stacked" }, written.map(([index, actions]) => element("span", {},
                element("span", { className: "faint" }, `${axis} ${index}: `), actions.join(", "))))
            : element("span", { className: "faint" }, "empty")]);
    }
    for (const [key, value] of Object.entries(route)) {
        if (!["dst", "mcast", "x", "y"].includes(key)) entries.push([key, renderPlainValue(value, key)]);
    }
    return keyValues(entries);
}

function describePathShape(path) {
    if (!path?.length) return "none";
    const childCounts = path.map((_, index) => path.filter((node) => node.parent === index).length);
    const nodes = `${path.length} node${path.length === 1 ? "" : "s"}`;
    return childCounts.some((count) => count > 1)
        ? `a tree of ${nodes}, ${childCounts.filter((count) => count === 0).length} branch ends` : `a chain of ${nodes}`;
}

function formatAction(action) {
    if (action === null || action === undefined) return element("span", { className: "faint" }, "stop");
    return Array.isArray(action) ? (action.length ? action.join(", ") : "empty") : String(action);
}

// What the hop's router does that the path doesn't act on: directions in the hop's action that aren't live there,
// and forwarding disabled on its receiver. `aside` says why a node has no router to look at.
function describeHopRouter(model, node) {
    const router = node.router ? model.routersByPath.get(node.router) : null;
    if (!router) {
        const chip = node.chip ? model.chipsByPath.get(node.chip) : null;
        let note = "no chip";
        if (node.router) note = "router not loaded";
        else if (chip) note = "on another rank: decode names only its own rank's routers";
        return { router: null, notes: [], aside: note };
    }
    const notes = [];
    const directions = router.data.analysis?.directions?.find((entry) => entry.vc === node.vc);
    if (Array.isArray(node.action) && directions?.live) {
        const notLive = node.action.filter((direction) => PLANAR_DIRECTIONS.includes(direction) && !directions.live.includes(direction));
        if (notLive.length) notes.push(`${notLive.join(", ")} not live here`);
    }
    for (const channel of Object.values(router.data.channels?.receivers?.[`vc${node.vc}`] ?? {})) {
        const disabled = Object.values(channel.fields ?? {}).some((fields) => fields?.forwarding_disabled?.value === true);
        if (disabled) { notes.push("forwarding disabled: delivers without reading the route"); break; }
    }
    return { router, notes, aside: null };
}

// One line per node, under the decoded file's key names, and below it what the node's router is doing: its verdict
// and anything describeHopRouter found. A list rather than a table, so it fits a channel card.
function renderPathList(model, path, onSelectRouter) {
    const term = (name, value) => element("span", { className: "hop-term" }, element("span", { className: "faint" }, `${name} `), value);
    return element("ol", { className: "hop-list", start: 0 }, path.map((node, index) => {
        const { router, notes, aside } = describeHopRouter(model, node);
        const verdict = router?.data.analysis?.activity?.verdict;
        const where = router ? routerLink(node.router, onSelectRouter) : element("span", {}, node.router ?? node.chip ?? "no chip");
        const problem = node.end && node.end !== "delivered";
        return element("li", { className: problem ? "path-problem" : "" },
            element("span", { className: "hop-index" }, `#${index}`),
            element("div", { className: "hop-body" },
                element("div", { className: "hop-line" },
                    element("b", {}, where),
                    term("vc", node.vc === null || node.vc === undefined ? "—" : String(node.vc)),
                    node.parent === null ? term("parent", "none") : term("from", `#${node.parent} via ${node.via ?? "?"}`),
                    node.reads ? term("reads", node.reads) : null,
                    term("action", formatAction(node.action)),
                    node.delivers ? element("span", { className: "hop-tag" }, "delivers") : null,
                    node.end ? element("span", { className: problem ? "hop-tag problem" : "hop-tag", title: PATH_END_NOTES[node.end] ?? "" },
                        `end: ${node.end.replaceAll("_", " ")}`) : null),
                element("div", { className: "hop-notes" },
                    verdict ? element("span", { className: `verdict-dot verdict-${verdict}` }) : null,
                    verdict ? ` ${verdict.replaceAll("_", " ")}` : "",
                    aside ? element("span", { className: "faint" }, aside) : null,
                    notes.length ? element("span", { className: "hop-warning" }, `${verdict ? " · " : ""}${notes.join(" · ")}`) : null)));
    }));
}

function routerLink(path, onSelectRouter) {
    const button = element("button", { type: "button", className: "link-button" }, shortRouterPath(path));
    button.addEventListener("click", () => onSelectRouter(path));
    return button;
}

// Where a channel's headers go, counted over every slot with one: the router that reads each packet next, what it
// does there, and where the packet goes after it.
export function renderHeaderSummary(model, headers, onSelectRouter) {
    const written = headers.filter(Boolean);
    if (!written.length) return null;
    const groups = new Map();
    for (const header of written) {
        const path = header.path ?? [];
        const first = path[0];
        const children = path.filter((node) => node.parent === 0);
        const then = children.length
            ? children.map((node) => shortRouterPath(node.router) || node.chip || node.end?.replaceAll("_", " ") || "stop").join(", ")
            : first?.end?.replaceAll("_", " ") ?? "—";
        const action = first ? (Array.isArray(first.action) ? first.action.join(", ") : first.action ?? "stop") : "—";
        const key = `${first?.router ?? first?.chip}|${action}|${then}`;
        if (!groups.has(key)) groups.set(key, { first, action, then, count: 0 });
        groups.get(key).count += 1;
    }
    const rows = [...groups.values()].sort((first, second) => second.count - first.count);
    return element("div", { className: "header-summary" },
        element("p", { className: "column-title" }, "Where the headers go"),
        element("table", { className: "data-table summary-table" },
            element("thead", {}, element("tr", {}, ["Slots", "Read next by", "Action there", "Then"].map((name) => element("th", { scope: "col" }, name)))),
            element("tbody", {}, rows.map((row) => element("tr", {},
                element("td", {}, String(row.count)),
                element("td", {}, row.first?.router && model.routersByPath.has(row.first.router)
                    ? routerLink(row.first.router, onSelectRouter) : shortRouterPath(row.first?.router) || row.first?.chip || "—"),
                element("td", {}, row.action),
                element("td", {}, row.then))))),
        element("p", { className: "hint" },
            `${written.length} of ${headers.length} slots have a header; the rest were never written. Each is the last header written to its slot.`));
}
