// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hardware view: every mesh of the capture on one zoomable canvas, chips placed by mesh_coord and coloured by
// decode's router verdicts, with one line per pair of linked chips. Wrap links go around the outside of their mesh.
// A slot's packet path can be drawn over it, hop by hop.

import { DIRECTION_NAMES } from "./capture_model.js";
import { element, escapeXml, formatCount } from "./format.js";
import { attachZoomPan } from "./zoom_pan.js";

// Worst first: a chip takes the colour of the worst verdict among its routers.
export const VERDICT_ORDER = ["stuck", "not_looping", "unknown", "looping", "running"];
export const VERDICT_DESCRIPTIONS = {
    stuck: "the loop runs, but a side can't make progress",
    not_looping: "the loop heartbeat didn't move",
    unknown: "too few reads to tell",
    looping: "the loop runs; no telemetry to tell running from stuck",
    running: "the loop and every telemetry heartbeat moved",
};

const CHIP_SIZE = 64;
const CHIP_PITCH = 112;
const LANE_START = 16;
const LANE_GAP = 9;
const MESH_GAP = 120;
const MESH_LABEL_SPACE = 30;
const STUB_LENGTH = 30;

// Builds the view in `panel` for `model`. onSelectChip(chipPath | null) is called when a click or key changes the
// selection.
export function createHardwareView(panel, model, { onSelectChip }) {
    const allChipLinks = buildChipLinks(model, null);
    const layout = layoutTopology(model, allChipLinks);
    const planes = [...new Set([...model.routersByPath.values()].map((router) => router.plane))]
        .filter((plane) => plane !== null).sort((first, second) => first - second);
    const hasRowWraps = allChipLinks.some((link) => link.wrapAxis === "row");
    const hasColumnWraps = allChipLinks.some((link) => link.wrapAxis === "column");
    const viewState = { plane: null, wrapsShown: "both", selectedChipPath: null, hoveredChipPath: null, shownPath: null };

    const planeControl = element("div", { className: "segmented", role: "group", ariaLabel: "Routing plane" });
    const wrapControl = element("div", { className: "segmented", role: "group", ariaLabel: "Wrap links" });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "topology");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${formatCount(model.chipsByPath.size, "chip")} in ${formatCount(model.meshes.size, "mesh", "meshes")}`);
    const readout = element("div", { className: "readout" });
    const legend = element("div", { className: "legend" });
    const pathBanner = element("div", { className: "path-banner", hidden: true });

    panel.replaceChildren(
        element("h2", { className: "panel-title" }, "Hardware"),
        element("p", { className: "blurb" }, describeTopology(model, planes)),
        element("div", { className: "toolbar" },
            planes.length > 1 ? element("div", { className: "options" }, "Routing plane", planeControl) : null,
            hasRowWraps || hasColumnWraps ? element("div", { className: "options" }, "Wrap links", wrapControl) : null),
        pathBanner,
        // The readout sits beside the stage: below it, its height changed with every hover, which moved the page under
        // the pointer when it was scrolled to the bottom.
        element("div", { className: "stage-row" },
            element("div", { className: "stage" },
                svg,
                element("div", { className: "zoom-controls" },
                    zoomButton("+", "Zoom in", () => zoomPan.zoomBy(1 / 1.4)),
                    zoomButton("−", "Zoom out", () => zoomPan.zoomBy(1.4)),
                    zoomButton("Fit", "Fit the whole topology", () => zoomPan.fit()))),
            readout),
        legend,
    );
    const zoomPan = attachZoomPan(svg, layout.bounds);

    function render() {
        const chipLinks = viewState.plane === null ? allChipLinks : buildChipLinks(model, viewState.plane);
        svg.innerHTML = drawTopology(model, layout, chipLinks, viewState);
        svg.classList.toggle("hide-row-wraps", viewState.wrapsShown === "columns" || viewState.wrapsShown === "none");
        svg.classList.toggle("hide-column-wraps", viewState.wrapsShown === "rows" || viewState.wrapsShown === "none");
        renderSegmented(planeControl, [["all", "All"], ...planes.map((plane) => [String(plane), String(plane)])],
            viewState.plane === null ? "all" : String(viewState.plane),
            (value) => { viewState.plane = value === "all" ? null : Number(value); render(); });
        renderSegmented(wrapControl, [["both", "Both"], ["rows", "Rows"], ["columns", "Columns"], ["none", "None"]]
            .filter(([value]) => (value !== "rows" || hasRowWraps) && (value !== "columns" || hasColumnWraps)),
            viewState.wrapsShown, (value) => { viewState.wrapsShown = value; render(); });
        renderLegend(legend, model, chipLinks, viewState);
        highlightChip(viewState.hoveredChipPath ?? viewState.selectedChipPath);
        drawShownPath();
    }

    function drawShownPath() {
        const shown = viewState.shownPath;
        svg.classList.toggle("showing-path", Boolean(shown));
        const { markup, chipPaths } = shown ? drawPacketPath(model, layout, shown) : { markup: "", chipPaths: new Set() };
        svg.querySelector(".overlays").innerHTML = markup;
        for (const chipElement of svg.querySelectorAll(".chip")) chipElement.classList.toggle("on-path", chipPaths.has(chipElement.dataset.path));
        pathBanner.hidden = !shown;
        if (shown) renderPathBanner(pathBanner, shown, () => showPath(null));
    }

    // Draws `shown` ({ label, routerPath, kind, path, onBack }) over the topology, or clears it with null.
    function showPath(shown) {
        viewState.shownPath = shown;
        drawShownPath();
    }

    function highlightChip(chipPath) {
        svg.classList.toggle("focusing", chipPath !== null);
        for (const linkElement of svg.querySelectorAll(".link")) {
            linkElement.classList.toggle("on", chipPath !== null && linkElement.dataset.chips.split(" ").includes(chipPath));
        }
        for (const chipElement of svg.querySelectorAll(".chip")) {
            chipElement.classList.toggle("selected", chipElement.dataset.path === viewState.selectedChipPath);
            chipElement.classList.toggle("hovered", chipElement.dataset.path === viewState.hoveredChipPath);
        }
        const chip = chipPath ? model.chipsByPath.get(chipPath) : null;
        if (chip) renderChipReadout(readout, model, chip, viewState.plane, chipPath === viewState.selectedChipPath);
        else readout.replaceChildren(element("p", { className: "hint" },
            "Hover a chip for its details, and click it to open it below. Scroll or pinch to zoom, and drag to pan."));
    }

    // Shows `chipPath` as selected without calling onSelectChip, for a selection made elsewhere.
    function showSelectedChip(chipPath) {
        viewState.selectedChipPath = chipPath;
        highlightChip(viewState.hoveredChipPath ?? chipPath);
    }

    function selectChip(chipPath) {
        showSelectedChip(chipPath);
        onSelectChip?.(chipPath);
    }

    const chipPathAt = (target) => target.closest?.(".chip")?.dataset.path ?? null;
    svg.addEventListener("pointerover", (event) => {
        const chipPath = chipPathAt(event.target);
        if (chipPath === viewState.hoveredChipPath) return;
        viewState.hoveredChipPath = chipPath;
        highlightChip(chipPath ?? viewState.selectedChipPath);
    });
    svg.addEventListener("pointerleave", () => {
        viewState.hoveredChipPath = null;
        highlightChip(viewState.selectedChipPath);
    });
    svg.addEventListener("click", (event) => {
        const chipPath = chipPathAt(event.target);
        selectChip(chipPath === viewState.selectedChipPath ? null : chipPath);
    });
    svg.addEventListener("keydown", (event) => {
        const chipPath = chipPathAt(event.target);
        if (chipPath && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            selectChip(chipPath === viewState.selectedChipPath ? null : chipPath);
        } else if (event.key === "Escape") {
            selectChip(null);
        }
    });
    svg.addEventListener("focusin", (event) => highlightChip(chipPathAt(event.target) ?? viewState.selectedChipPath));

    render();
    return { showSelectedChip, showPath, detach: () => zoomPan.detach() };
}

function zoomButton(text, label, onClick) {
    const button = element("button", { type: "button", className: "ghost", title: label, ariaLabel: label }, text);
    button.addEventListener("click", onClick);
    return button;
}

function renderSegmented(container, options, selectedValue, onChange) {
    container.replaceChildren(...options.map(([value, text]) => {
        const button = element("button", { type: "button" }, text);
        button.setAttribute("aria-pressed", String(value === selectedValue));
        button.addEventListener("click", () => onChange(value));
        return button;
    }));
}

function describeTopology(model, planes) {
    const meshes = [...model.meshes.values()].map((mesh) => {
        const wrappedAxes = Object.entries(mesh.torus ?? {}).filter(([, wraps]) => wraps).map(([axis]) => axis).sort();
        return `${mesh.id} is ${mesh.rows} × ${mesh.columns} chips (rows × columns)` +
            (wrappedAxes.length ? ` and wraps on ${wrappedAxes.join(" and ")}` : " and doesn't wrap");
    });
    const hostRanks = new Set([...model.chipsByPath.values()].map((chip) => chip.hostRank));
    return `${model.fabric.config?.fabric_config ?? "Unknown config"}, topology ${model.fabric.topology}. ` +
        `${meshes.join("; ")}. ${formatCount(planes.length, "routing plane")}, ` +
        `with chips on ${formatCount(hostRanks.size, "host rank")}.`;
}

// ---------- Links between chips ----------

// One entry per pair of linked chips and kind of link, from the router links on `plane` (every plane when null).
// kind is cardinal, wrap, express, intermesh or z. A link whose peer chip isn't in the capture at all is a stub.
function buildChipLinks(model, plane) {
    const chipLinks = new Map();
    for (const link of model.links) {
        const [router, peer] = link.ends;
        if (plane !== null && router.plane !== plane) continue;
        const fromChip = router.chip;
        const toChip = peer?.chip ?? model.chipsByPath.get(link.endPaths[1].split("/").slice(0, 2).join("/")) ?? null;
        const kind = classifyLink(router, fromChip, toChip, link);
        const key = toChip ? `${[fromChip.path, toChip.path].sort().join(" ")} ${kind}` : `${router.path} stub`;
        if (!chipLinks.has(key)) {
            chipLinks.set(key, {
                kind,
                fromChip,
                toChip,
                direction: router.direction,
                wrapAxis: kind === "wrap" ? findWrapAxis(fromChip, toChip) : null,
                routerLinks: [],
                planes: new Set(),
                crossRank: false,
                confirmed: true,
            });
        }
        const chipLink = chipLinks.get(key);
        chipLink.routerLinks.push(link);
        if (router.plane !== null) chipLink.planes.add(router.plane);
        chipLink.crossRank ||= link.crossRank;
        chipLink.confirmed &&= link.check === "both_ends";
    }
    return [...chipLinks.values()];
}

function classifyLink(router, fromChip, toChip, link) {
    const edgeCapability = router.data.link?.edge_capability;
    if (edgeCapability === "intermesh" || (toChip && toChip.mesh !== fromChip.mesh)) return "intermesh";
    if (router.direction === "Z") return "z";
    if (edgeCapability === "intramesh_express") return "express";
    return link.wrap ? "wrap" : "cardinal";
}

// A wrap joins the two ends of a row or of a column.
function findWrapAxis(fromChip, toChip) {
    if (!toChip) return null;
    if (fromChip.row === toChip.row) return "row";
    if (fromChip.column === toChip.column) return "column";
    return null;
}

// ---------- Layout ----------

// Places the meshes side by side. Each row and column that has a wrap link gets its own lane around its mesh, so
// wraps don't run on top of each other. Rows nearer the top and columns nearer the left go further out.
function layoutTopology(model, chipLinks) {
    const meshes = [...model.meshes.values()].sort((first, second) => compareMeshIds(first.id, second.id));
    const meshLayouts = new Map();
    let nextX = 0;
    let height = 0;
    for (const mesh of meshes) {
        const meshLinks = chipLinks.filter((link) => link.fromChip.mesh === mesh && link.kind === "wrap");
        const wrapRows = [...new Set(meshLinks.filter((link) => link.wrapAxis === "row").map((link) => link.fromChip.row))].sort((a, b) => a - b);
        const wrapColumns = [...new Set(meshLinks.filter((link) => link.wrapAxis === "column").map((link) => link.fromChip.column))].sort((a, b) => a - b);

        const rowLaneDepth = wrapRows.length ? LANE_START + (wrapRows.length - 1) * LANE_GAP : 0;
        const columnTopDepth = wrapColumns.length ? LANE_START + (wrapColumns.length - 1) * LANE_GAP : 0;
        const columnOuterStart = rowLaneDepth + LANE_START;
        const columnOuterDepth = wrapColumns.length ? columnOuterStart + (wrapColumns.length - 1) * LANE_GAP : 0;
        const rowLanes = new Map(wrapRows.map((row, index) => [row, LANE_START + (wrapRows.length - 1 - index) * LANE_GAP]));
        const columnLanes = new Map(wrapColumns.map((column, index) => [column, {
            top: LANE_START + (wrapColumns.length - 1 - index) * LANE_GAP,
            outer: columnOuterStart + (wrapColumns.length - 1 - index) * LANE_GAP,
        }]));

        const outside = Math.max(rowLaneDepth, columnOuterDepth) + STUB_LENGTH;
        const padding = { left: rowLaneDepth + STUB_LENGTH, top: MESH_LABEL_SPACE + columnTopDepth + STUB_LENGTH, right: outside, bottom: outside };
        const gridWidth = (mesh.columns - 1) * CHIP_PITCH + CHIP_SIZE;
        const gridHeight = (mesh.rows - 1) * CHIP_PITCH + CHIP_SIZE;
        meshLayouts.set(mesh, {
            originX: nextX + padding.left,
            originY: padding.top,
            gridWidth,
            gridHeight,
            rowLanes,
            columnLanes,
            labelX: nextX + padding.left,
            labelY: MESH_LABEL_SPACE - 10,
        });
        nextX += padding.left + gridWidth + padding.right + MESH_GAP;
        height = Math.max(height, padding.top + gridHeight + padding.bottom);
    }
    return { meshLayouts, bounds: { x: 0, y: 0, width: Math.max(nextX - MESH_GAP, 1), height: Math.max(height, 1) } };
}

function compareMeshIds(first, second) {
    return Number(first.slice(1)) - Number(second.slice(1)) || first.localeCompare(second);
}

function chipBox(layout, chip) {
    const meshLayout = layout.meshLayouts.get(chip.mesh);
    const x = meshLayout.originX + chip.column * CHIP_PITCH;
    const y = meshLayout.originY + chip.row * CHIP_PITCH;
    return { x, y, centerX: x + CHIP_SIZE / 2, centerY: y + CHIP_SIZE / 2 };
}

// ---------- Drawing ----------

function drawTopology(model, layout, chipLinks, viewState) {
    const planeCounts = chipLinks.filter((link) => link.toChip).map((link) => link.planes.size);
    const usualPlaneCount = mostCommon(planeCounts);
    let linksSvg = "";
    let labelsSvg = "";
    for (const link of chipLinks) {
        const classes = ["link", link.kind];
        if (link.wrapAxis) classes.push(`${link.wrapAxis}-wrap`);
        if (link.crossRank) classes.push("cross-rank");
        if (!link.confirmed) classes.push("unconfirmed");
        if (!link.toChip) classes.push("stub");
        const chips = [link.fromChip.path, link.toChip?.path].filter(Boolean).join(" ");
        const { path, labelAt } = drawLinkPath(layout, link);
        linksSvg += `<path class="${classes.join(" ")}" data-chips="${escapeXml(chips)}" d="${path}"/>`;
        // Planes are labelled only where a pair of chips has a different number of them than most pairs.
        if (labelAt && link.toChip && link.planes.size !== usualPlaneCount) {
            labelsSvg += `<text class="link-label" x="${labelAt.x}" y="${labelAt.y}">${link.planes.size}</text>`;
        }
    }

    let meshLabelsSvg = "";
    for (const [mesh, meshLayout] of layout.meshLayouts) {
        meshLabelsSvg += `<text class="mesh-label" x="${meshLayout.labelX}" y="${meshLayout.labelY}">${escapeXml(mesh.id)} · ${mesh.rows} × ${mesh.columns}</text>`;
    }

    const findingsByChipPath = countFindingsByChip(model, viewState.plane);
    let chipsSvg = "";
    for (const chip of model.chipsByPath.values()) chipsSvg += drawChip(layout, chip, viewState.plane, findingsByChipPath.get(chip.path) ?? 0);

    return `<g class="links">${linksSvg}</g><g class="link-labels">${labelsSvg}</g><g class="mesh-labels">${meshLabelsSvg}</g>` +
        `<g class="chips">${chipsSvg}</g><g class="overlays"></g>`;
}

function drawLinkPath(layout, link) {
    const from = chipBox(layout, link.fromChip);
    if (!link.toChip) {
        // A stub out of the chip's side in the router's direction: its peer isn't in the capture.
        const [dx, dy] = { N: [0, -1], S: [0, 1], E: [1, 0], W: [-1, 0] }[link.direction] ?? [1, -1];
        const startX = from.centerX + (dx * CHIP_SIZE) / 2;
        const startY = from.centerY + (dy * CHIP_SIZE) / 2;
        return { path: `M${startX},${startY} L${startX + dx * STUB_LENGTH},${startY + dy * STUB_LENGTH}`, labelAt: null };
    }
    const to = chipBox(layout, link.toChip);
    if (link.kind === "wrap" && link.wrapAxis) return { path: drawWrapPath(layout, link, from, to), labelAt: null };
    if (link.kind === "cardinal") {
        return {
            path: `M${from.centerX},${from.centerY} L${to.centerX},${to.centerY}`,
            labelAt: { x: (from.centerX + to.centerX) / 2 + 6, y: (from.centerY + to.centerY) / 2 - 6 },
        };
    }
    // Express, Z and intermesh links curve, so they stand apart from the grid's straight lines.
    const middleX = (from.centerX + to.centerX) / 2;
    const middleY = (from.centerY + to.centerY) / 2;
    const length = Math.hypot(to.centerX - from.centerX, to.centerY - from.centerY) || 1;
    const bend = Math.min(60, length / 4);
    const controlX = middleX - ((to.centerY - from.centerY) / length) * bend;
    const controlY = middleY + ((to.centerX - from.centerX) / length) * bend;
    return {
        path: `M${from.centerX},${from.centerY} Q${controlX},${controlY} ${to.centerX},${to.centerY}`,
        labelAt: { x: (middleX + controlX) / 2, y: (middleY + controlY) / 2 },
    };
}

// A row's wrap leaves its last chip to the right, runs down below the mesh and back up the left side into its first
// chip. A column's wrap leaves its top chip upward, runs right, down the right side, under the mesh and up into its
// bottom chip.
function drawWrapPath(layout, link, from, to) {
    return roundedPath(wrapPoints(layout.meshLayouts.get(link.fromChip.mesh), link.wrapAxis, link.fromChip, from, to), 10);
}

// The corners of a wrap between two chip boxes of a row or column, in the order a packet takes them: a row's from its
// last chip eastward round to its first, a column's from its top chip northward round to its bottom.
function wrapPoints(meshLayout, axis, chip, from, to) {
    const left = meshLayout.originX;
    const top = meshLayout.originY;
    const right = left + meshLayout.gridWidth;
    const bottom = top + meshLayout.gridHeight;
    if (axis === "row") {
        const [first, last] = from.x < to.x ? [from, to] : [to, from];
        const depth = meshLayout.rowLanes.get(chip.row) ?? LANE_START;
        const y = first.centerY;
        return [[last.x + CHIP_SIZE, y], [right + depth, y], [right + depth, bottom + depth],
            [left - depth, bottom + depth], [left - depth, y], [first.x, y]];
    }
    const [upper, lower] = from.y < to.y ? [from, to] : [to, from];
    const lane = meshLayout.columnLanes.get(chip.column) ?? { top: LANE_START, outer: LANE_START };
    const x = upper.centerX;
    return [[x, upper.y], [x, top - lane.top], [right + lane.outer, top - lane.top],
        [right + lane.outer, bottom + lane.outer], [x, bottom + lane.outer], [x, lower.y + CHIP_SIZE]];
}

function roundedPath(points, radius) {
    let path = `M${points[0][0]},${points[0][1]}`;
    for (let index = 1; index < points.length - 1; index++) {
        const [previousX, previousY] = points[index - 1];
        const [x, y] = points[index];
        const [nextX, nextY] = points[index + 1];
        const inLength = Math.hypot(x - previousX, y - previousY) || 1;
        const outLength = Math.hypot(nextX - x, nextY - y) || 1;
        const cornerRadius = Math.min(radius, inLength / 2, outLength / 2);
        const startX = x - ((x - previousX) / inLength) * cornerRadius;
        const startY = y - ((y - previousY) / inLength) * cornerRadius;
        const endX = x + ((nextX - x) / outLength) * cornerRadius;
        const endY = y + ((nextY - y) / outLength) * cornerRadius;
        path += ` L${startX},${startY} Q${x},${y} ${endX},${endY}`;
    }
    const [lastX, lastY] = points.at(-1);
    return `${path} L${lastX},${lastY}`;
}

// ---------- A packet's path ----------

const DIRECTION_VECTORS = { east: [1, 0], west: [-1, 0], north: [0, -1], south: [0, 1] };
const DIRECTION_WORDS = { N: "north", E: "east", S: "south", W: "west", Z: "z" };
const PATH_END_LABELS = { no_action: "no action", off_mesh: "off mesh", unknown_chip: "unknown chip", loops: "loops", leaves_mesh: "leaves mesh" };

// The path's hops as arrows from each node's parent chip to its own, its chips marked: the slot's chip outlined, a
// chip the packet is delivered to dotted, and a branch that ends without delivering flagged with why. A sender's slot
// hasn't crossed its link yet, so a dashed arrow joins the slot's chip to the receiver that reads it next.
function drawPacketPath(model, layout, shown) {
    const path = shown.path ?? [];
    const chipOf = (chipPath) => (chipPath ? model.chipsByPath.get(chipPath) ?? null : null);
    const startRouter = model.routersByPath.get(shown.routerPath) ?? null;
    const chipPaths = new Set();
    let arrows = "";
    let marks = "";
    if (startRouter) {
        chipPaths.add(startRouter.chip.path);
        const box = chipBox(layout, startRouter.chip);
        marks += `<rect class="path-start" x="${box.x - 6}" y="${box.y - 6}" width="${CHIP_SIZE + 12}" height="${CHIP_SIZE + 12}" rx="10"/>`;
        const firstChip = chipOf(path[0]?.chip);
        if (shown.kind === "sender" && firstChip && firstChip !== startRouter.chip) {
            arrows += drawHopArrow(layout, startRouter.chip, firstChip, DIRECTION_WORDS[startRouter.direction], "path-hop pending");
        }
    }
    path.forEach((node, index) => {
        const chip = chipOf(node.chip);
        const parentChip = node.parent === null ? null : chipOf(path[node.parent]?.chip);
        if (parentChip && chip) {
            arrows += drawHopArrow(layout, parentChip, chip, node.via, "path-hop");
        } else if (parentChip) {
            // A stop with no chip: an arrow out of the parent's side, ending in the reason.
            const box = chipBox(layout, parentChip);
            const [dx, dy] = DIRECTION_VECTORS[node.via] ?? [1, -1];
            const startX = box.centerX + (dx * CHIP_SIZE) / 2;
            const startY = box.centerY + (dy * CHIP_SIZE) / 2;
            const endX = startX + dx * (STUB_LENGTH + 6);
            const endY = startY + dy * (STUB_LENGTH + 6);
            arrows += `<path class="path-hop problem" d="M${startX},${startY} L${endX},${endY}"><title>#${index} ${escapeXml(node.end ?? "stop")}</title></path>`;
            marks += drawPathEnd(endX + dx * 10, endY + dy * 10, node.end, index);
        }
        if (!chip) return;
        chipPaths.add(chip.path);
        const box = chipBox(layout, chip);
        if (node.end && node.end !== "delivered") {
            marks += drawPathEnd(box.x, box.y, node.end, index);
        } else if (node.delivers) {
            marks += `<circle class="path-delivers" cx="${box.x}" cy="${box.y}" r="6"><title>#${index} delivers to ${escapeXml(chip.path)}</title></circle>`;
        }
    });
    const defs = `<defs><marker id="path-arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
        `<path class="path-arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>`;
    return { markup: `${defs}<g class="path-arrows">${arrows}</g><g class="path-marks">${marks}</g>`, chipPaths };
}

function drawPathEnd(x, y, end, index) {
    const label = PATH_END_LABELS[end] ?? end ?? "stop";
    return `<g class="path-end"><circle cx="${x}" cy="${y}" r="8"/><text x="${x}" y="${y + 3.5}">!</text>` +
        `<text class="path-end-label" x="${x}" y="${y - 13}">${escapeXml(label)}</text><title>#${index}: ${escapeXml(label)}</title></g>`;
}

// One hop, from a chip box's edge to the next one's. A hop to a neighbouring chip is straight; a hop the long way
// along a row or column, as a wrap takes, follows that wrap's lane round the mesh; anything else curves.
function drawHopArrow(layout, fromChip, toChip, via, className) {
    if (fromChip === toChip) return "";
    const from = chipBox(layout, fromChip);
    const to = chipBox(layout, toChip);
    const marker = `marker-end="url(#path-arrowhead)"`;
    const rowStep = toChip.row - fromChip.row;
    const columnStep = toChip.column - fromChip.column;
    const sameMesh = fromChip.mesh === toChip.mesh;
    const half = CHIP_SIZE / 2;
    if (sameMesh && Math.abs(rowStep) + Math.abs(columnStep) === 1) {
        const startX = from.centerX + columnStep * half;
        const startY = from.centerY + rowStep * half;
        const endX = to.centerX - columnStep * (half + 2);
        const endY = to.centerY - rowStep * (half + 2);
        return `<path class="${className}" d="M${startX},${startY} L${endX},${endY}" ${marker}/>`;
    }
    const [dx, dy] = DIRECTION_VECTORS[via] ?? [0, 0];
    const meshLayout = layout.meshLayouts.get(fromChip.mesh);
    if (sameMesh && rowStep === 0 && dx !== 0 && Math.sign(columnStep) !== dx) {
        const points = wrapPoints(meshLayout, "row", fromChip, from, to);
        return `<path class="${className}" d="${roundedPath(dx > 0 ? points : points.reverse(), 10)}" ${marker}/>`;
    }
    if (sameMesh && columnStep === 0 && dy !== 0 && Math.sign(rowStep) !== dy) {
        const points = wrapPoints(meshLayout, "column", fromChip, from, to);
        return `<path class="${className}" d="${roundedPath(dy < 0 ? points : points.reverse(), 10)}" ${marker}/>`;
    }
    const length = Math.hypot(to.centerX - from.centerX, to.centerY - from.centerY) || 1;
    const bend = Math.min(60, length / 4);
    const controlX = (from.centerX + to.centerX) / 2 - ((to.centerY - from.centerY) / length) * bend;
    const controlY = (from.centerY + to.centerY) / 2 + ((to.centerX - from.centerX) / length) * bend;
    const towardEnd = Math.hypot(to.centerX - controlX, to.centerY - controlY) || 1;
    const endX = to.centerX - ((to.centerX - controlX) / towardEnd) * (half + 2);
    const endY = to.centerY - ((to.centerY - controlY) / towardEnd) * (half + 2);
    return `<path class="${className}" d="M${from.centerX},${from.centerY} Q${controlX},${controlY} ${endX},${endY}" ${marker}/>`;
}

function renderPathBanner(banner, shown, onClear) {
    const clearButton = element("button", { type: "button", className: "ghost" }, "Clear");
    clearButton.addEventListener("click", onClear);
    const backButton = shown.onBack ? element("button", { type: "button", className: "ghost" }, "Back to the slot") : null;
    backButton?.addEventListener("click", shown.onBack);
    const swatch = (className) => {
        const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        icon.innerHTML = className === "path-start"
            ? `<rect class="path-start" x="4" y="0" width="22" height="12" rx="3"/>`
            : className === "path-delivers" ? `<circle class="path-delivers" cx="15" cy="6" r="5"/>`
                : `<path class="${className}" d="M2,6 L28,6"/>`;
        return icon;
    };
    const ends = new Set((shown.path ?? []).map((node) => node.end).filter((end) => end && end !== "delivered"));
    banner.replaceChildren(
        element("div", { className: "path-banner-head" },
            element("span", {}, "Path of ", element("b", {}, shown.label)),
            element("span", { className: "path-banner-buttons" }, backButton, clearButton)),
        element("div", { className: "legend" },
            element("span", {}, swatch("path-start"), "the slot's chip"),
            element("span", {}, swatch("path-hop"), "a hop the route takes"),
            shown.kind === "sender" ? element("span", {}, swatch("path-hop pending"), "not sent yet: across the link to the next receiver") : null,
            element("span", {}, swatch("path-delivers"), "delivers to the chip"),
            ends.size ? element("span", { className: "path-problem-note" }, `! ends without delivering: ${[...ends].map((end) => PATH_END_LABELS[end] ?? end).join(", ")}`) : null));
}

function drawChip(layout, chip, plane, findingCount) {
    const box = chipBox(layout, chip);
    const routers = routersOnPlane(chip, plane);
    const loaded = chip.ownerRankNumber !== null;
    const counts = countVerdicts(routers);
    const worst = VERDICT_ORDER.find((verdict) => counts.has(verdict)) ?? null;
    const allRead = routers.every((router) => router.data.analysis?.capture?.status === "complete");
    const classes = ["chip", loaded ? `verdict-${worst ?? "none"}` : "not-loaded"];
    if (!allRead) classes.push("partly-read");
    const status = !loaded ? "not loaded" : worst ? `${counts.get(worst)} ${worst.replace("_", " ")}` : "no routers";
    const label = `${chip.path}, row ${chip.row}, column ${chip.column}: ${status}`;
    let badge = "";
    if (findingCount > 0) {
        badge = `<circle class="finding-badge" cx="${box.x + CHIP_SIZE}" cy="${box.y}" r="9"/>` +
            `<text class="finding-count" x="${box.x + CHIP_SIZE}" y="${box.y + 3.5}">${findingCount}</text>`;
    }
    return `<g class="${classes.join(" ")}" data-path="${escapeXml(chip.path)}" tabindex="0" role="button" aria-label="${escapeXml(label)}">` +
        `<rect x="${box.x}" y="${box.y}" width="${CHIP_SIZE}" height="${CHIP_SIZE}" rx="7"/>` +
        `<text class="chip-id" x="${box.centerX}" y="${box.y + 24}">${escapeXml(chip.id)}</text>` +
        `<text class="chip-coord" x="${box.centerX}" y="${box.y + 38}">${chip.row}, ${chip.column}</text>` +
        `<text class="chip-status" x="${box.centerX}" y="${box.y + 54}">${escapeXml(status)}</text>${badge}</g>`;
}

function routersOnPlane(chip, plane) {
    return plane === null ? chip.routers : chip.routers.filter((router) => router.plane === plane);
}

function countVerdicts(routers) {
    const counts = new Map();
    for (const router of routers) {
        const verdict = router.data.analysis?.activity?.verdict ?? "unknown";
        counts.set(verdict, (counts.get(verdict) ?? 0) + 1);
    }
    return counts;
}

// Each finding counts once on every chip one of its paths is on, when one of those paths is on `plane`.
function countFindingsByChip(model, plane) {
    const counts = new Map();
    for (const finding of model.findings) {
        const chipPaths = new Set();
        for (const path of finding.paths ?? []) {
            const router = model.routersByPath.get(path);
            if (plane !== null && router && router.plane !== plane) continue;
            chipPaths.add(path.split("/").slice(0, 2).join("/"));
        }
        for (const chipPath of chipPaths) counts.set(chipPath, (counts.get(chipPath) ?? 0) + 1);
    }
    return counts;
}

function mostCommon(values) {
    const counts = new Map();
    for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
    return [...counts].sort((first, second) => second[1] - first[1])[0]?.[0] ?? null;
}

// ---------- Legend and readout ----------

function renderLegend(legend, model, chipLinks, viewState) {
    const routers = viewState.plane === null
        ? [...model.routersByPath.values()]
        : [...model.routersByPath.values()].filter((router) => router.plane === viewState.plane);
    const counts = countVerdicts(routers);
    const swatch = (className) => {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "0 0 30 12");
        svg.innerHTML = className.startsWith("chip")
            ? `<g class="${className}"><rect x="8" y="0.5" width="14" height="11" rx="2"/></g>`
            : `<path class="link ${className}" d="M2,6 L28,6"/>`;
        return svg;
    };
    const items = VERDICT_ORDER.filter((verdict) => counts.has(verdict)).map((verdict) =>
        element("span", { title: VERDICT_DESCRIPTIONS[verdict] }, swatch(`chip verdict-${verdict}`),
            `${verdict.replace("_", " ")} (${counts.get(verdict)} routers)`));
    const kinds = new Set(chipLinks.map((link) => link.kind));
    const linkItems = [
        ["cardinal", "Link", kinds.has("cardinal")],
        ["wrap", "Wrap link", kinds.has("wrap")],
        ["express", "Express link", kinds.has("express")],
        ["z", "Z link", kinds.has("z")],
        ["intermesh", "Intermesh link", kinds.has("intermesh")],
        ["cardinal cross-rank", "Crosses MPI ranks", chipLinks.some((link) => link.crossRank)],
        ["cardinal unconfirmed", "Not confirmed from both ends", chipLinks.some((link) => !link.confirmed)],
    ].filter(([, , present]) => present).map(([className, text]) => element("span", {}, swatch(className), text));
    const extras = [];
    if (model.findings.length) extras.push(element("span", {}, element("span", { className: "badge-sample" }, "n"), "Findings on the chip"));
    if ([...model.routersByPath.values()].some((router) => router.data.analysis?.capture?.status !== "complete")) {
        extras.push(element("span", {}, swatch("chip partly-read verdict-none"), "Not every router read"));
    }
    if ([...model.chipsByPath.values()].some((chip) => chip.ownerRankNumber === null)) {
        extras.push(element("span", {}, swatch("chip not-loaded"), "Rank not loaded"));
    }
    legend.replaceChildren(...items, ...linkItems, ...extras);
}

function renderChipReadout(readout, model, chip, plane, pinned) {
    const title = element("div", { className: "readout-title" },
        element("b", {}, chip.path), ` · row ${chip.row}, column ${chip.column}`,
        pinned ? element("span", { className: "pill" }, "selected") : null);
    if (chip.ownerRankNumber === null) {
        readout.replaceChildren(title, element("p", { className: "hint" },
            `Host rank ${chip.hostRank} owns this chip, and its decoded file isn't loaded, so its routers aren't known.`));
        return;
    }
    const ownerRank = model.ranks.find((rank) => rank.number === chip.ownerRankNumber);
    const routers = routersOnPlane(chip, plane);
    const counts = countVerdicts(routers);
    const facts = [
        ["Physical chip", String(chip.data.physical_chip_id ?? "unknown")],
        ["ASIC", chip.data.asic_id ?? "unknown"],
        ["Host", `rank ${chip.hostRank} on ${ownerRank?.decoded.run.host_name ?? "unknown host"}`],
        ["Verdicts", VERDICT_ORDER.filter((verdict) => counts.has(verdict))
            .map((verdict) => `${counts.get(verdict)} ${verdict.replace("_", " ")}`).join(", ") || "no routers"],
    ];
    const notRunning = routers.filter((router) => ["stuck", "not_looping", "unknown"].includes(router.data.analysis?.activity?.verdict));
    if (notRunning.length) {
        facts.push(["Not running", notRunning.map((router) => {
            const activity = router.data.analysis.activity;
            const stopped = (activity.stopped ?? []).map((side) => `erisc${side.erisc} ${side.side}`).join(", ");
            return `${router.name} ${activity.verdict.replace("_", " ")}${stopped ? ` (${stopped} stopped)` : ""}`;
        }).join("; ")]);
    }
    const notRead = routers.filter((router) => router.data.analysis?.capture?.status !== "complete");
    if (notRead.length) {
        facts.push(["Not fully read", notRead.map((router) => `${router.name} ${router.data.analysis?.capture?.status ?? "unknown"}`).join(", ")]);
    }
    facts.push(["Links", describeChipLinks(model, routers)]);
    readout.replaceChildren(title, element("dl", {}, facts.map(([label, value]) =>
        element("div", {}, element("dt", {}, label), element("dd", {}, value)))));
}

// "N → M0/C28 (planes 0–3, wrap, crosses ranks)", one line per direction and peer chip.
function describeChipLinks(model, routers) {
    const byDirectionAndPeer = new Map();
    for (const router of routers) {
        const link = model.linkByRouterPath.get(router.path);
        const peerPath = router.data.link?.peer;
        if (!peerPath) continue;
        const peerChipPath = peerPath.split("/").slice(0, 2).join("/");
        const key = `${router.direction} ${peerChipPath}`;
        if (!byDirectionAndPeer.has(key)) {
            byDirectionAndPeer.set(key, { direction: router.direction, peerChipPath, planes: [], wrap: false, crossRank: false, confirmed: true });
        }
        const entry = byDirectionAndPeer.get(key);
        entry.planes.push(router.plane);
        entry.wrap ||= Boolean(router.data.link.wrap);
        entry.crossRank ||= Boolean(router.data.link.cross_host);
        entry.confirmed &&= link?.check === "both_ends";
    }
    if (byDirectionAndPeer.size === 0) return "none";
    return element("span", { className: "link-lines" }, [...byDirectionAndPeer.values()].map((entry) => {
        const notes = [`plane${entry.planes.length === 1 ? "" : "s"} ${entry.planes.sort((a, b) => a - b).join(", ")}`];
        if (entry.wrap) notes.push("wrap");
        if (entry.crossRank) notes.push("crosses ranks");
        if (!entry.confirmed) notes.push("not confirmed from both ends");
        const directionName = DIRECTION_NAMES[entry.direction] ?? entry.direction ?? "?";
        return element("span", {}, `${directionName} → ${entry.peerChipPath} (${notes.join(", ")})`);
    }));
}
