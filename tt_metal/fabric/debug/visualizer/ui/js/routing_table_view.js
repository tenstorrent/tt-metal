// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A router's routing table, routing_l1_info_t, as decode interprets it, shown as text. state_manager is the router's
// own; everything else is the chip's, written once per copy in the chip's analysis.routing, so every router that
// shares a copy shows the same tables. Only the parts this fabric config fills are shown.

import { element, formatCount } from "./format.js";

const ARROWS = { east: "→", west: "←", north: "↑", south: "↓", z: "⊙" };

function routerButton(path, onSelectRouter, text) {
    const button = element("button", { type: "button", className: "link-button" }, text);
    button.addEventListener("click", () => onSelectRouter(path));
    return button;
}

// A direction by name, with an arrow for the four planar ones. The kernel's magic values read as faint.
function renderDirection(direction) {
    if (!direction) return element("span", { className: "faint" }, "—");
    if (direction === "stop") return element("span", { className: "faint" }, "stop");
    if (!(direction in ARROWS)) return element("span", { className: "faint" }, direction);
    return element("span", { className: `direction direction-${direction}` }, `${ARROWS[direction]} ${direction}`);
}

function section(title, hint, ...children) {
    return element("div", { className: "routing-part" },
        element("p", { className: "column-title" }, title),
        hint ? element("p", { className: "hint" }, hint) : null,
        ...children);
}

export function renderRoutingTable(model, router, onSelectRouter) {
    const area = router.data.l1_areas?.routing_table ?? null;
    const routing = router.chip.data.analysis?.routing ?? null;
    const copyIndex = router.data.analysis?.routing_table?.copy;
    const copy = copyIndex === undefined || copyIndex === null ? null : routing?.copies?.[copyIndex] ?? null;
    const parts = [renderRouterPart(model, router, area, routing, copyIndex, copy, onSelectRouter)];
    if (copy?.table) {
        const table = copy.table;
        parts.push(renderIdentity(router, table.identity));
        if (table.intra_mesh_directions) parts.push(renderIntraMesh(table.intra_mesh_directions, table.identity?.device_id));
        if (table.inter_mesh_directions || table.exit_nodes) parts.push(renderInterMesh(table.inter_mesh_directions, table.exit_nodes, table.identity?.mesh_id));
        if (table.path_table_1d) parts.push(renderPathTable1d(table.path_table_1d));
        if (table.action_vectors) parts.push(renderActionVectors(table.action_vectors, table.identity?.mesh_coord));
        if (table.multicast_trees) parts.push(renderMulticastTrees(table.multicast_trees));
    }
    return element("section", { className: "router-section routing-table" },
        element("h3", { className: "section-title" }, "Routing table"),
        element("p", { className: "hint" }, "routing_l1_info_t, as decode reads it. state_manager is this router's own; the rest is the chip's, the same on every router that holds this copy."),
        ...parts);
}

function renderRouterPart(model, router, area, routing, copyIndex, copy, onSelectRouter) {
    const state = area?.decoded?.value ?? null;
    const rows = [];
    rows.push(["state_manager", state
        ? element("span", {}, `state ${state.state?.name ?? state.state?.value ?? "?"}, command ${state.command?.name ?? state.command?.value ?? "?"}`)
        : element("span", { className: "faint" }, area ? "not read" : "no routing_table area")]);
    if (copy) {
        const copies = routing.copies.length;
        rows.push(["copy", `${copyIndex}${copies > 1 ? ` of ${copies}: the chip's routers don't all hold the same table` : ", the chip's only one"}`]);
        rows.push(["read from", routerButton(copy.read_router, onSelectRouter, copy.read_router.split("/").at(-1))]);
        const siblings = copy.routers.map((path) => path.split("/").at(-1));
        rows.push(["held by", `${formatCount(siblings.length, "router")}: ${siblings.join(", ")}`]);
    } else {
        rows.push(["copy", element("span", { className: "faint" }, "none: this router's table wasn't read")]);
    }
    const chipRouterPaths = new Set(router.chip.routers.map((chipRouter) => chipRouter.path));
    const findings = model.findings.filter((finding) => finding.code?.startsWith("routing_table")
        && (finding.paths ?? []).some((path) => path === router.chip.path || chipRouterPaths.has(path)));
    return section("This router", null,
        element("dl", { className: "key-values routing-facts" }, rows.map(([key, value]) =>
            element("div", {}, element("dt", {}, key), element("dd", {}, value)))),
        findings.length ? element("ul", { className: "routing-findings" }, findings.map((finding) =>
            element("li", { className: `severity-${finding.severity}` }, element("b", {}, finding.code), ` · ${finding.message ?? ""}`))) : null);
}

// The table's identity beside what the manifest says the chip is.
function renderIdentity(router, identity) {
    if (!identity) return null;
    const chip = router.chip;
    const expected = {
        mesh_id: Number(chip.mesh.id.slice(1)),
        device_id: Number(chip.id.slice(1)),
        mesh_coord: [chip.row, chip.column],
        mesh_shape: [chip.mesh.rows, chip.mesh.columns],
    };
    const show = (value) => (Array.isArray(value) ? `[${value.join(", ")}]` : String(value));
    const rows = Object.entries(identity).map(([key, value]) => {
        const chipValue = expected[key];
        const same = chipValue === undefined || show(chipValue) === show(value);
        // 1D builds don't write the mesh shape or coordinates; they read as zeros.
        const unwritten = !same && Array.isArray(value) && value.every((item) => item === 0);
        let note = "";
        if (!same) note = unwritten ? "zero: not written in 1D" : "differs from the chip";
        return element("tr", { className: !same && !unwritten ? "mismatch" : "" },
            element("th", { scope: "row" }, key), element("td", {}, show(value)),
            element("td", {}, chipValue === undefined ? "—" : show(chipValue)),
            element("td", { className: "faint" }, note));
    });
    return section("Identity", null, element("table", { className: "data-table routing-identity" },
        element("thead", {}, element("tr", {}, ["Member", "In the table", "The chip", ""].map((name) => element("th", { scope: "col" }, name)))),
        element("tbody", {}, rows)));
}

// The first hop toward each chip of the mesh, one row per destination.
function renderIntraMesh(directions, ownId) {
    return section("intra_mesh_directions",
        "The direction a packet from this chip leaves by toward each chip of the mesh: what a worker's get_next_hop_router_direction returns. invalid_direction is the chip's own entry, or a destination with no route.",
        element("table", { className: "data-table routing-entries" },
            element("thead", {}, element("tr", {}, element("th", { scope: "col" }, "Destination"), element("th", { scope: "col" }, "First hop"))),
            element("tbody", {}, directions.map((direction, id) => element("tr", {},
                element("th", { scope: "row" }, `C${id}`),
                element("td", {}, renderDirection(direction), id === ownId ? element("span", { className: "faint" }, " · this chip") : null))))));
}

function renderInterMesh(directions, exitNodes, ownMeshId) {
    const meshIds = [...new Set([...Object.keys(directions ?? {}), ...Object.keys(exitNodes ?? {})])].sort((first, second) => first - second);
    return section("Across meshes",
        "inter_mesh_directions: the direction toward each destination mesh. exit_nodes (2D): the chip of this mesh a packet leaves from toward it. Entries the host didn't fill are left out.",
        element("table", { className: "data-table" },
            element("thead", {}, element("tr", {}, ["Mesh", "inter_mesh_directions", ...(exitNodes ? ["exit_nodes"] : [])].map((name) => element("th", { scope: "col" }, name)))),
            element("tbody", {}, meshIds.map((meshId) => element("tr", {},
                element("th", { scope: "row" }, `M${meshId}`, Number(meshId) === ownMeshId ? element("span", { className: "faint" }, " · this mesh") : null),
                element("td", {}, renderDirection(directions?.[meshId])),
                exitNodes ? element("td", {}, exitNodes[meshId] === undefined ? element("span", { className: "faint" }, "—") : `C${exitNodes[meshId]}`) : null)))));
}

// Runs of one action collapsed: forward_only × 3, write_only.
function compactActions(actions) {
    const runs = [];
    for (const action of actions) {
        if (runs.at(-1)?.[0] === action) runs.at(-1)[1] += 1;
        else runs.push([action, 1]);
    }
    return runs.map(([action, count]) => (count > 1 ? `${action} × ${count}` : action)).join(", ");
}

// Each nonzero entry: its index (the distance to its destination), hops (how many actions it holds, at most what its
// words fit) and its actions.
function renderPathTable1d(pathTable) {
    return section("path_table_1d",
        `The hop actions for a unicast to a chip index hops away, hop 0 first. hops is how many actions the entry holds, at most what its words fit. ${formatCount(pathTable.num_zero_entries ?? 0, "entry", "entries")} are zero, including index 0, the route to the chip itself.`,
        element("table", { className: "data-table" },
            element("thead", {}, element("tr", {}, ["index", "hops", "actions"].map((name) => element("th", { scope: "col" }, name)))),
            element("tbody", {}, (pathTable.entries ?? []).map((entry) => element("tr", {},
                element("th", { scope: "row" }, String(entry.index ?? "?")),
                element("td", {}, String(entry.hops)),
                element("td", { className: "wrap-cell", title: entry.actions.join(", ") }, compactActions(entry.actions)))))));
}

// y[dst_y][cur_y] and x[dst_x][cur_x]: one row per destination, one column per coordinate the packet is at. This
// chip's own column is highlighted: the step it takes itself.
function renderActionVectors(actionVectors, meshCoord) {
    const [ownRow, ownColumn] = meshCoord ?? [];
    const matrix = (name, rows, axis, ownIndex) => {
        if (!rows) return element("p", { className: "hint" }, `${name}: null (the table's shape isn't valid)`);
        const width = Math.max(0, ...rows.map((row) => row.length));
        return element("div", { className: "table-wrap" }, element("table", { className: "data-table action-matrix" },
            element("thead", {}, element("tr", {}, element("th", { scope: "col" }, `${name}: to ${axis} ↓ · at ${axis} →`),
                Array.from({ length: width }, (_, index) => element("th", { scope: "col", className: index === ownIndex ? "own" : "" }, String(index))))),
            element("tbody", {}, rows.map((row, destination) => element("tr", {},
                element("th", { scope: "row" }, `${axis} ${destination}`),
                row.map((code, current) => element("td", { className: current === ownIndex ? "own" : "" }, renderDirection(code))))))));
    };
    return section("action_vectors",
        "The 2D unicast route. Row: the destination's row (y) or column (x). Column: the row or column the packet is at. Each cell is the way it goes from there; stop means it has reached the destination's row (y, then it follows x) or column (x). The same on every chip of the mesh; this chip's own row and column are highlighted.",
        matrix("y", actionVectors.y, "row", ownRow),
        matrix("x", actionVectors.x, "column", ownColumn));
}

function renderMulticastTrees(trees) {
    const tree = (name, axis, edges) => element("div", {},
        element("p", { className: "subtle-title" }, `${name}: ${formatCount(edges?.length ?? 0, "edge")}`),
        edges?.length
            ? element("table", { className: "data-table" },
                element("thead", {}, element("tr", {}, ["parent", "direction", "child"].map((key) => element("th", { scope: "col" }, key)))),
                element("tbody", {}, edges.map((edge) => element("tr", {},
                    element("td", {}, `${axis} ${edge.parent}`), element("td", {}, renderDirection(edge.direction)), element("td", {}, `${axis} ${edge.child}`)))))
            : element("p", { className: "hint" }, "none"));
    return section("multicast_trees",
        "This chip's reverse multicast trees, in stored order: to reach a child row (y) or column (x), its parent sends the packet this way.",
        element("div", { className: "tree-columns" }, tree("y", "row", trees.y), tree("x", "column", trees.x)));
}
