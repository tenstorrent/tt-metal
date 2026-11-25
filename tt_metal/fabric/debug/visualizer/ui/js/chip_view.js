// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The chip view: one chip's die with its routers on the side each one faces, outlined by decode's verdict. Inside the
// die, arrows show where each router's receiver forwards on the selected plane (intra_chip_downstream_edges), solid
// when the router's live_downstreams includes that direction (always in 1D, where live_downstreams is null). Below it, a table of every router.

import { DIRECTION_NAMES } from "./capture_model.js";
import { element, escapeXml, formatCount } from "./format.js";
import { VERDICT_DESCRIPTIONS, VERDICT_ORDER } from "./hardware_view.js";

const DIE = { x: 230, y: 180, width: 360, height: 360 };
const DIE_CENTER = { x: DIE.x + DIE.width / 2, y: DIE.y + DIE.height / 2 };
const ROUTER_WIDTH = 48;
const ROUTER_HEIGHT = 36;
const ROUTER_GAP = 4;
const STUB_LENGTH = 30;
const PEER_LABEL_DEPTH = 40;
const VIEW_BOX = "84 44 652 632";
// live_downstreams and analysis.directions name directions in lower case.
const DIRECTION_WORDS = { N: "north", E: "east", S: "south", W: "west", Z: "z" };

// Builds the view in `panel`. onSelectChip(chipPath) is called to show another chip (the dropdown, a peer label or
// Close, which passes null); onSelectRouter(routerPath | null) when a click or key selects a router.
export function createChipView(panel, model, { onSelectChip, onSelectRouter }) {
    const viewState = { chip: null, plane: null, vc: null, selectedRouterPath: null, hovered: null };
    let edges = [];

    const title = element("h2", { className: "panel-title" }, "Chip");
    const closeButton = element("button", { type: "button", className: "ghost" }, "Close");
    closeButton.addEventListener("click", () => onSelectChip(null));
    const chipSelect = buildChipSelect(model);
    chipSelect.setAttribute("aria-label", "Chip");
    chipSelect.addEventListener("change", () => onSelectChip(chipSelect.value));
    const planePicker = element("div", { className: "plane-picker", role: "group", ariaLabel: "Routing plane" });
    const vcControl = element("div", { className: "segmented", role: "group", ariaLabel: "Virtual channel" });
    const vcOption = element("div", { className: "options" }, "Edges on", vcControl);
    const facts = element("dl", { className: "facts chip-facts" });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "die-view");
    svg.setAttribute("viewBox", VIEW_BOX);
    svg.setAttribute("role", "img");
    const readout = element("div", { className: "readout" });
    const legend = element("div", { className: "legend" });
    const tableBody = element("tbody");
    const dieArea = element("div", { className: "die-area" },
        // Beside the die, so hovering, which changes the readout, never changes the page's height.
        element("div", { className: "stage-row" }, element("div", { className: "stage die-stage" }, svg), readout), legend,
        element("div", { className: "table-wrap" }, element("table", { className: "router-table" },
            element("thead", {}, element("tr", {}, ["Router", "Direction", "Plane", "Eth channel", "Peer", "Link",
                "Verdict", "Capture", "Findings"].map((heading) => element("th", { scope: "col" }, heading)))),
            tableBody)));
    const notLoadedNote = element("p", { className: "hint" });

    panel.replaceChildren(
        element("div", { className: "panel-head" }, title, closeButton),
        element("div", { className: "toolbar" },
            element("label", { className: "options" }, "Chip", chipSelect), planePicker, vcOption),
        facts, notLoadedNote, dieArea);

    function showChip(chipPath) {
        const chip = chipPath ? model.chipsByPath.get(chipPath) ?? null : null;
        panel.hidden = !chip;
        if (!chip) return;
        if (chip !== viewState.chip) {
            viewState.chip = chip;
            viewState.selectedRouterPath = null;
            viewState.hovered = null;
            const planes = planesOf(chip);
            if (!planes.includes(viewState.plane)) viewState.plane = planes[0] ?? null;
            viewState.vc = null;
        }
        render();
    }

    function render() {
        const chip = viewState.chip;
        // The arrows are redrawn, so an arrow's index may now name another one; the next pointer move sets it again.
        if (viewState.hovered?.kind === "edge") viewState.hovered = null;
        title.textContent = `Chip ${chip.path}`;
        chipSelect.value = chip.path;
        renderFacts(facts, model, chip);
        const loaded = chip.ownerRankNumber !== null;
        dieArea.hidden = !loaded;
        notLoadedNote.hidden = loaded;
        notLoadedNote.textContent = loaded ? "" :
            `Host rank ${chip.hostRank} owns this chip, and its decoded file isn't loaded, so its routers aren't known.`;
        const planes = planesOf(chip);
        planePicker.replaceChildren(...(planes.length > 1 ? ["Routing plane", ...planes.map((plane) => {
            const button = element("button", { type: "button", className: "plane-button" },
                element("span", { className: "swatch", style: { background: planeColor(plane) } }), `Plane ${plane}`);
            button.setAttribute("aria-pressed", String(plane === viewState.plane));
            button.addEventListener("click", () => { viewState.plane = plane; viewState.vc = null; render(); });
            return button;
        })] : []));
        if (!loaded) return;

        const vcs = edgeVcsOn(chip, viewState.plane);
        if (!vcs.includes(viewState.vc)) viewState.vc = vcs[0] ?? null;
        vcOption.hidden = vcs.length < 2;
        vcControl.replaceChildren(...vcs.map((vc) => {
            const button = element("button", { type: "button" }, vc.toUpperCase());
            button.setAttribute("aria-pressed", String(vc === viewState.vc));
            button.addEventListener("click", () => { viewState.vc = vc; render(); });
            return button;
        }));

        const placed = placeRouters(chip);
        edges = findEdges(model, chip, viewState.plane, viewState.vc);
        svg.setAttribute("aria-label", `${chip.path} and its ${formatCount(chip.routers.length, "router")}`);
        svg.innerHTML = drawDie(model, chip, placed, edges, viewState);
        renderTable(tableBody, model, chip, viewState, selectRouter);
        renderLegend(legend, model, chip, edges);
        renderReadout();
    }

    function renderReadout() {
        const hovered = viewState.hovered;
        if (hovered?.kind === "edge") {
            renderEdgeReadout(readout, edges[hovered.index]);
        } else if (hovered?.kind === "router" || viewState.selectedRouterPath) {
            const routerPath = hovered?.kind === "router" ? hovered.path : viewState.selectedRouterPath;
            renderRouterReadout(readout, model, model.routersByPath.get(routerPath), routerPath === viewState.selectedRouterPath);
        } else {
            readout.replaceChildren(element("p", { className: "hint" },
                "Hover a router or an arrow for its details, and click a router to select it. Click a side's label to open the chip it links to."));
        }
        for (const routerElement of svg.querySelectorAll(".router")) {
            routerElement.classList.toggle("hovered", hovered?.kind === "router" && routerElement.dataset.path === hovered.path);
        }
        for (const edgeElement of svg.querySelectorAll(".edge")) {
            edgeElement.classList.toggle("hovered", hovered?.kind === "edge" && Number(edgeElement.dataset.index) === hovered.index);
        }
    }

    // Shows `routerPath` (on the chip on show) as selected, switching to its plane, without calling onSelectRouter.
    function showSelectedRouter(routerPath) {
        const router = routerPath ? model.routersByPath.get(routerPath) : null;
        viewState.selectedRouterPath = router?.chip === viewState.chip ? routerPath : null;
        if (viewState.selectedRouterPath && router.plane !== null && router.plane !== viewState.plane) {
            viewState.plane = router.plane;
            viewState.vc = null;
        }
        if (viewState.chip) render();
    }

    function selectRouter(routerPath) {
        showSelectedRouter(routerPath);
        onSelectRouter?.(viewState.selectedRouterPath);
    }

    svg.addEventListener("pointerover", (event) => {
        const routerElement = event.target.closest?.(".router");
        const edgeElement = event.target.closest?.(".edge");
        const peerElement = event.target.closest?.(".peer");
        if (routerElement) viewState.hovered = { kind: "router", path: routerElement.dataset.path };
        else if (edgeElement) viewState.hovered = { kind: "edge", index: Number(edgeElement.dataset.index) };
        else if (!peerElement) viewState.hovered = null;
        renderReadout();
    });
    svg.addEventListener("pointerleave", () => { viewState.hovered = null; renderReadout(); });
    svg.addEventListener("click", (event) => {
        const routerPath = event.target.closest?.(".router")?.dataset.path;
        const peerChipPath = event.target.closest?.(".peer")?.dataset.chip;
        if (routerPath) selectRouter(routerPath === viewState.selectedRouterPath ? null : routerPath);
        else if (peerChipPath) onSelectChip(peerChipPath);
    });
    svg.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        const routerPath = event.target.closest?.(".router")?.dataset.path;
        const peerChipPath = event.target.closest?.(".peer")?.dataset.chip;
        if (!routerPath && !peerChipPath) return;
        event.preventDefault();
        if (routerPath) selectRouter(routerPath === viewState.selectedRouterPath ? null : routerPath);
        else onSelectChip(peerChipPath);
    });
    svg.addEventListener("focusin", (event) => {
        const routerPath = event.target.closest?.(".router")?.dataset.path;
        viewState.hovered = routerPath ? { kind: "router", path: routerPath } : null;
        renderReadout();
    });

    panel.hidden = true;
    return { showChip, showSelectedRouter };
}

function planeColor(plane) {
    return plane === null ? "var(--fg-muted)" : `var(--plane-${plane % 4})`;
}

function planesOf(chip) {
    return [...new Set(chip.routers.map((router) => router.plane))].filter((plane) => plane !== null).sort((a, b) => a - b);
}

// Every chip, by mesh: "C0 · 0, 0 · physical 0". With more than one mesh, each mesh is an option group.
function buildChipSelect(model) {
    const chipOption = (chip) => {
        const detail = chip.ownerRankNumber === null ? "rank not loaded" : `physical ${chip.data.physical_chip_id ?? "?"}`;
        return element("option", { value: chip.path }, `${chip.id} · ${chip.row}, ${chip.column} · ${detail}`);
    };
    const meshes = [...model.meshes.values()];
    if (meshes.length === 1) return element("select", {}, meshes[0].chips.map(chipOption));
    return element("select", {}, meshes.map((mesh) => element("optgroup", { label: mesh.id }, mesh.chips.map(chipOption))));
}

function renderFacts(facts, model, chip) {
    const ownerRank = model.ranks.find((rank) => rank.number === chip.ownerRankNumber);
    const directions = new Set(chip.routers.map((router) => router.direction));
    const planes = planesOf(chip);
    const localSync = chip.data.local_sync;
    const entries = [
        ["Mesh coordinates", `row ${chip.row}, column ${chip.column}`],
        ["Physical chip", String(chip.data.physical_chip_id ?? "unknown")],
        ["ASIC", chip.data.asic_id ?? "unknown"],
        ["Host", ownerRank ? `rank ${chip.hostRank} on ${ownerRank.decoded.run.host_name ?? "unknown host"}` : `rank ${chip.hostRank}, not loaded`],
    ];
    if (chip.ownerRankNumber !== null) {
        entries.push(["Routers", `${chip.routers.length} on ${formatCount(planes.length, "plane")}, ${formatCount(directions.size, "direction")}`]);
        if (localSync) {
            entries.push(["Local sync master", `${localSync.master?.split("/").at(-1) ?? "none"}, syncing ${formatCount(localSync.num_routers ?? 0, "router")}`]);
        }
    }
    if (chip.data.z_port_role && chip.data.z_port_role !== "none") entries.push(["Z port role", chip.data.z_port_role]);
    facts.replaceChildren(...entries.map(([label, value]) => element("div", {}, element("dt", {}, label), element("dd", {}, value))));
}

// ---------- Layout ----------

// Each router's box, the point on the die's edge its arrows start from, and its stub out to the peer label. Routers
// on a side are ordered by plane: left to right on N and S, top to bottom on E and W. Z routers (and any router whose
// direction isn't known) sit in a row inside the die, near its bottom.
function placeRouters(chip) {
    const sides = { N: [], E: [], S: [], W: [], Z: [] };
    for (const router of chip.routers) sides[router.direction in sides ? router.direction : "Z"].push(router);
    const placed = new Map();
    for (const [side, routers] of Object.entries(sides)) {
        const count = routers.length;
        const along = side === "E" || side === "W" ? DIE.height : DIE.width;
        const pitch = Math.min(side === "E" || side === "W" ? 58 : 74, (along - 24) / Math.max(count, 1));
        routers.forEach((router, index) => {
            const offset = (index - (count - 1) / 2) * pitch;
            let box;
            let anchor;
            let stub = null;
            if (side === "N") {
                const x = DIE_CENTER.x + offset;
                box = { x: x - ROUTER_WIDTH / 2, y: DIE.y - ROUTER_GAP - ROUTER_HEIGHT };
                anchor = { x, y: DIE.y };
                stub = [x, box.y, x, box.y - STUB_LENGTH];
            } else if (side === "S") {
                const x = DIE_CENTER.x + offset;
                box = { x: x - ROUTER_WIDTH / 2, y: DIE.y + DIE.height + ROUTER_GAP };
                anchor = { x, y: DIE.y + DIE.height };
                stub = [x, box.y + ROUTER_HEIGHT, x, box.y + ROUTER_HEIGHT + STUB_LENGTH];
            } else if (side === "E") {
                const y = DIE_CENTER.y + offset;
                box = { x: DIE.x + DIE.width + ROUTER_GAP, y: y - ROUTER_HEIGHT / 2 };
                anchor = { x: DIE.x + DIE.width, y };
                stub = [box.x + ROUTER_WIDTH, y, box.x + ROUTER_WIDTH + STUB_LENGTH, y];
            } else if (side === "W") {
                const y = DIE_CENTER.y + offset;
                box = { x: DIE.x - ROUTER_GAP - ROUTER_WIDTH, y: y - ROUTER_HEIGHT / 2 };
                anchor = { x: DIE.x, y };
                stub = [box.x, y, box.x - STUB_LENGTH, y];
            } else {
                const x = DIE_CENTER.x + offset;
                box = { x: x - ROUTER_WIDTH / 2, y: DIE.y + DIE.height - ROUTER_HEIGHT - 20 };
                anchor = { x, y: box.y };
            }
            placed.set(router.path, { router, side, box, anchor, stub });
        });
    }
    return placed;
}

// ---------- Edges inside the chip ----------

function edgeVcsOn(chip, plane) {
    const vcs = new Set();
    for (const router of chip.routers) {
        if (plane !== null && router.plane !== plane) continue;
        for (const [vc, vcEdges] of Object.entries(router.data.intra_chip_downstream_edges ?? {})) {
            if (Object.keys(vcEdges ?? {}).length) vcs.add(vc);
        }
    }
    return [...vcs].sort((first, second) => Number(first.slice(2)) - Number(second.slice(2)));
}

// Every edge from a router on `plane` to the sender channel it forwards into, on `vc`. An edge is live when the
// router's live_downstreams for that VC names the downstream router's direction.
function findEdges(model, chip, plane, vc) {
    if (vc === null) return [];
    const found = [];
    for (const router of chip.routers) {
        if (plane !== null && router.plane !== plane) continue;
        // In 1D live_downstreams is null: a packet's route only says whether to forward, to the one downstream.
        const live = router.data.live_downstreams === null || router.data.live_downstreams === undefined ? null : router.data.live_downstreams[vc] ?? [];
        for (const [edgeName, edge] of Object.entries(router.data.intra_chip_downstream_edges?.[vc] ?? {})) {
            const channelPath = edge.downstream_channel ?? "";
            const [downstreamPath, channel] = channelPath.split("/channels/");
            const downstream = model.routersByPath.get(downstreamPath) ?? null;
            const freeSlots = Object.values(edge.fields ?? {}).map((fields) => fields.free_slots?.decoded).find(Boolean) ?? null;
            found.push({
                router,
                downstream,
                downstreamPath,
                channel: channel ?? channelPath,
                edgeName,
                vc,
                live: live === null || (downstream ? live.includes(DIRECTION_WORDS[downstream.direction]) : false),
                liveKnown: live !== null,
                throughTensixMux: Boolean(edge.through_tensix_mux),
                freeSlots,
            });
        }
    }
    return found;
}

// ---------- Drawing ----------

function drawDie(model, chip, placed, edges, viewState) {
    const plane = viewState.plane;
    const markerIds = new Set();
    let edgesSvg = "";
    edges.forEach((edge, index) => {
        const from = placed.get(edge.router.path)?.anchor;
        const to = placed.get(edge.downstreamPath)?.anchor;
        if (!from || !to) return;
        const markerId = `arrow-${edge.live ? "live" : "idle"}-${edge.router.plane ?? "x"}`;
        markerIds.add(markerId);
        const path = curvedEdge(from, to);
        const classes = ["edge", edge.live ? "live" : "idle"];
        if (edge.throughTensixMux) classes.push("through-mux");
        edgesSvg += `<g class="${classes.join(" ")}" data-index="${index}" style="--plane:${planeColor(edge.router.plane)}">` +
            `<path class="edge-hit" d="${path}"/><path class="edge-line" d="${path}" marker-end="url(#${markerId})"/></g>`;
    });
    const markers = [...markerIds].map((markerId) => {
        const [, liveness, markerPlane] = markerId.split("-");
        return `<marker id="${markerId}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">` +
            `<path d="M0,0 L10,5 L0,10 z" class="arrowhead ${liveness}" style="fill:${planeColor(markerPlane === "x" ? null : Number(markerPlane))}"/></marker>`;
    }).join("");

    let anchorsSvg = "";
    let stubsSvg = "";
    let routersSvg = "";
    const skippedRouterPaths = new Set(model.discoverySkippedCores.map((skipped) => skipped.router?.path).filter(Boolean));
    for (const { router, box, anchor, stub } of placed.values()) {
        const onPlane = plane === null || router.plane === plane;
        const color = planeColor(router.plane);
        if (onPlane) anchorsSvg += `<circle class="anchor" cx="${anchor.x}" cy="${anchor.y}" r="3.5" style="fill:${color}"/>`;
        if (stub) stubsSvg += `<line class="router-stub${onPlane ? "" : " dim"}" x1="${stub[0]}" y1="${stub[1]}" x2="${stub[2]}" y2="${stub[3]}" style="stroke:${color}"/>`;
        const verdict = router.data.analysis?.activity?.verdict ?? "unknown";
        const classes = ["router", `verdict-${verdict}`];
        if (!onPlane) classes.push("dim");
        if (router.data.analysis?.capture?.status !== "complete") classes.push("partly-read");
        if (router.path === viewState.selectedRouterPath) classes.push("selected");
        const ethChannel = router.data.identity?.eth_chan;
        const label = `${router.name}, ${DIRECTION_NAMES[router.direction] ?? "unknown direction"}, plane ${router.plane}, ` +
            `Ethernet channel ${ethChannel ?? "unknown"}: ${verdict.replace("_", " ")}`;
        routersSvg += `<g class="${classes.join(" ")}" data-path="${escapeXml(router.path)}" tabindex="0" role="button" aria-label="${escapeXml(label)}">` +
            `<rect x="${box.x}" y="${box.y}" width="${ROUTER_WIDTH}" height="${ROUTER_HEIGHT}" rx="5"/>` +
            `<text class="router-name" x="${box.x + ROUTER_WIDTH / 2}" y="${box.y + 15}" style="fill:${color}">${escapeXml(router.name)}</text>` +
            `<text class="router-channel" x="${box.x + ROUTER_WIDTH / 2}" y="${box.y + 28}">eth ${ethChannel ?? "?"}</text>` +
            (skippedRouterPaths.has(router.path) ? `<circle class="skipped-mark" cx="${box.x + ROUTER_WIDTH}" cy="${box.y}" r="3.5"/>` : "") +
            `</g>`;
    }

    return `<defs>${markers}<pattern id="tensix-grid" width="22" height="22" patternUnits="userSpaceOnUse">` +
        `<rect class="die-grid" x="3" y="3" width="16" height="16" rx="2"/></pattern></defs>` +
        `<rect class="die" x="${DIE.x}" y="${DIE.y}" width="${DIE.width}" height="${DIE.height}" rx="10"/>` +
        `<rect x="${DIE.x + 8}" y="${DIE.y + 8}" width="${DIE.width - 16}" height="${DIE.height - 16}" rx="6" fill="url(#tensix-grid)"/>` +
        `<g class="edges">${edgesSvg}</g><g class="anchors">${anchorsSvg}</g>` +
        `<g class="stubs">${stubsSvg}</g><g class="routers">${routersSvg}</g>` +
        drawPeerLabels(model, chip, placed);
}

// A curve from one anchor to another, bent to the left of its direction of travel, so the edges A→B and B→A bend
// apart instead of lying on top of each other.
function curvedEdge(from, to) {
    const length = Math.hypot(to.x - from.x, to.y - from.y) || 1;
    const bend = Math.min(Math.max(length * 0.2, 18), 46);
    const controlX = (from.x + to.x) / 2 + ((to.y - from.y) / length) * bend;
    const controlY = (from.y + to.y) / 2 - ((to.x - from.x) / length) * bend;
    return `M${from.x},${from.y} Q${controlX},${controlY} ${to.x},${to.y}`;
}

// A label outside each side: where that side's routers link to. Clicking it opens the peer chip when there's one.
function drawPeerLabels(model, chip, placed) {
    const labels = {
        N: { x: DIE_CENTER.x, y: DIE.y - ROUTER_GAP - ROUTER_HEIGHT - STUB_LENGTH - PEER_LABEL_DEPTH / 2, rotate: 0 },
        S: { x: DIE_CENTER.x, y: DIE.y + DIE.height + ROUTER_GAP + ROUTER_HEIGHT + STUB_LENGTH + PEER_LABEL_DEPTH / 2, rotate: 0 },
        E: { x: DIE.x + DIE.width + ROUTER_GAP + ROUTER_WIDTH + STUB_LENGTH + PEER_LABEL_DEPTH / 2, y: DIE_CENTER.y, rotate: 90 },
        W: { x: DIE.x - ROUTER_GAP - ROUTER_WIDTH - STUB_LENGTH - PEER_LABEL_DEPTH / 2, y: DIE_CENTER.y, rotate: -90 },
    };
    let svg = "";
    for (const [side, position] of Object.entries(labels)) {
        const routers = [...placed.values()].filter((entry) => entry.side === side).map((entry) => entry.router);
        const description = describeSide(model, side, routers);
        const width = side === "E" || side === "W" ? DIE.height : DIE.width;
        const classes = ["peer"];
        if (!routers.length) classes.push("none");
        if (description.peerChipPath) classes.push("navigable");
        const focusable = description.peerChipPath
            ? ` tabindex="0" role="button" data-chip="${escapeXml(description.peerChipPath)}" aria-label="Open ${escapeXml(description.peerChipPath)}"`
            : "";
        svg += `<g class="${classes.join(" ")}"${focusable} transform="rotate(${position.rotate} ${position.x} ${position.y})">` +
            `<rect x="${position.x - width / 2}" y="${position.y - PEER_LABEL_DEPTH / 2}" width="${width}" height="${PEER_LABEL_DEPTH}" rx="5"/>` +
            `<text class="peer-main" x="${position.x}" y="${position.y - (description.detail ? 5 : -4)}">${escapeXml(description.main)}</text>` +
            (description.detail ? `<text class="peer-detail" x="${position.x}" y="${position.y + 12}">${escapeXml(description.detail)}</text>` : "") +
            `</g>`;
    }
    return `<g class="peers">${svg}</g>`;
}

// "EAST → M0/C1" and "W0–W3 · wrap · crosses ranks", from the routers' link.peer.
function describeSide(model, side, routers) {
    const sideName = (DIRECTION_NAMES[side] ?? side).toUpperCase();
    if (!routers.length) return { main: `${sideName} · no routers`, detail: "", peerChipPath: null };
    const peerPaths = routers.map((router) => router.data.link?.peer).filter(Boolean);
    if (!peerPaths.length) return { main: `${sideName} · no peer`, detail: "", peerChipPath: null };
    const peerChipPaths = [...new Set(peerPaths.map((path) => path.split("/").slice(0, 2).join("/")))];
    const notes = [compactRouterNames(peerPaths.map((path) => path.split("/").at(-1)))];
    if (routers.some((router) => router.data.link?.wrap)) notes.push("wrap");
    if (routers.some((router) => router.data.link?.cross_host)) notes.push("crosses ranks");
    if (routers.some((router) => model.linkByRouterPath.get(router.path)?.check !== "both_ends")) notes.push("not confirmed from both ends");
    if (!peerChipPaths.every((path) => model.chipsByPath.has(path))) notes.push("peer chip not in the capture");
    return {
        main: `${sideName} → ${peerChipPaths.join(", ")}`,
        detail: notes.join(" · "),
        peerChipPath: peerChipPaths.length === 1 && model.chipsByPath.has(peerChipPaths[0]) ? peerChipPaths[0] : null,
    };
}

// ["W0", "W1", "W2", "W3"] → "W0–W3"; anything else is listed.
function compactRouterNames(names) {
    const sorted = [...new Set(names)].sort();
    const parsed = sorted.map((name) => /^([NESWZ])(\d+)$/.exec(name));
    const sameDirection = parsed.every((match) => match && match[1] === parsed[0][1]);
    const planes = parsed.map((match) => Number(match?.[2]));
    const contiguous = planes.every((plane, index) => index === 0 || plane === planes[index - 1] + 1);
    if (sorted.length > 2 && sameDirection && contiguous) return `${sorted[0]}–${sorted.at(-1)}`;
    return sorted.join(", ");
}

// ---------- Table, legend and readout ----------

function renderTable(tableBody, model, chip, viewState, selectRouter) {
    const skippedRouterPaths = new Set(model.discoverySkippedCores.map((skipped) => skipped.router?.path).filter(Boolean));
    tableBody.replaceChildren(...chip.routers.map((router) => {
        const link = router.data.link ?? {};
        const linkNotes = [];
        if (link.edge_capability && link.edge_capability !== "intramesh_cardinal") linkNotes.push(link.edge_capability);
        if (link.wrap) linkNotes.push("wrap");
        if (link.cross_host) linkNotes.push("crosses ranks");
        if (link.peer && model.linkByRouterPath.get(router.path)?.check !== "both_ends") linkNotes.push("not confirmed");
        const activity = router.data.analysis?.activity ?? {};
        const verdict = activity.verdict ?? "unknown";
        const stopped = (activity.stopped ?? []).map((side) => `erisc${side.erisc} ${side.side}`).join(", ");
        const capture = router.data.analysis?.capture ?? {};
        const captureText = (capture.status ?? "unknown").replace("_", " ") + (skippedRouterPaths.has(router.path) ? ", discovery skipped" : "");
        const findings = findingsOn(model, router);
        const classes = [];
        if (router.path === viewState.selectedRouterPath) classes.push("selected");
        if (viewState.plane !== null && router.plane !== viewState.plane) classes.push("other-plane");
        const row = element("tr", { className: classes.join(" "), tabIndex: 0 },
        element("td", {}, router.name),
        element("td", {}, DIRECTION_NAMES[router.direction] ?? "?"),
        element("td", {}, String(router.plane ?? "?")),
        element("td", {}, String(router.data.identity?.eth_chan ?? "?")),
        element("td", {}, link.peer ?? "none"),
        element("td", {}, linkNotes.join(", ") || "—"),
        element("td", { title: VERDICT_DESCRIPTIONS[verdict] ?? "" },
            element("span", { className: `verdict-dot verdict-${verdict}` }), verdict.replace("_", " "), stopped ? ` (${stopped} stopped)` : ""),
        element("td", {}, captureText),
        element("td", { title: findings.map((finding) => `${finding.severity}: ${finding.code}: ${finding.message}`).join("\n") },
            findings.length ? String(findings.length) : "—"));
        row.style.setProperty("--plane", planeColor(router.plane));
        row.addEventListener("click", () => selectRouter(router.path === viewState.selectedRouterPath ? null : router.path));
        row.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                selectRouter(router.path === viewState.selectedRouterPath ? null : router.path);
            }
        });
        return row;
    }));
}

function findingsOn(model, router) {
    return model.findings.filter((finding) =>
        (finding.paths ?? []).some((path) => path === router.path || path.startsWith(`${router.path}/`)));
}

function renderLegend(legend, model, chip, edges) {
    const verdicts = new Set(chip.routers.map((router) => router.data.analysis?.activity?.verdict ?? "unknown"));
    const swatch = (markup) => {
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("viewBox", "0 0 30 12");
        svg.innerHTML = markup;
        return svg;
    };
    const items = VERDICT_ORDER.filter((verdict) => verdicts.has(verdict)).map((verdict) =>
        element("span", { title: VERDICT_DESCRIPTIONS[verdict] },
            swatch(`<g class="router verdict-${verdict}"><rect x="8" y="0.5" width="14" height="11" rx="2"/></g>`), verdict.replace("_", " ")));
    if (edges.some((edge) => edge.live)) {
        items.push(element("span", {}, swatch(`<g class="edge live" style="--plane:var(--fg-muted)"><path class="edge-line" d="M2,6 L28,6"/></g>`),
            "Live direction"));
    }
    if (edges.some((edge) => !edge.live)) {
        items.push(element("span", {}, swatch(`<g class="edge idle" style="--plane:var(--fg-muted)"><path class="edge-line" d="M2,6 L28,6"/></g>`),
            "Connected, not a live direction"));
    }
    if (chip.routers.some((router) => router.data.analysis?.capture?.status !== "complete")) {
        items.push(element("span", {}, swatch(`<g class="router partly-read"><rect x="8" y="0.5" width="14" height="11" rx="2"/></g>`), "Not fully read"));
    }
    if (model.discoverySkippedCores.some((skipped) => skipped.router?.chip === chip)) {
        items.push(element("span", {}, swatch(`<circle class="skipped-mark" cx="15" cy="6" r="3.5"/>`),
            "Skipped by UMD discovery (its heartbeat looked stalled)"));
    }
    legend.replaceChildren(...items);
}

function renderEdgeReadout(readout, edge) {
    const downstreamName = edge.downstream?.name ?? edge.downstreamPath;
    const facts = [
        ["From", `${edge.router.name} receiver, ${edge.vc}`],
        ["Into", `${downstreamName} ${edge.channel.replace("senders/", "sender ").replaceAll("/", " ")}`],
        ["Live direction", !edge.liveKnown ? "yes: in 1D the router forwards to its one downstream" :
            edge.live ? "yes: live_downstreams names this direction" : "no: connected, but live_downstreams doesn't name this direction"],
    ];
    if (edge.freeSlots) facts.push(["Free slots", `${edge.freeSlots.count} of ${edge.freeSlots.depth}`]);
    if (edge.throughTensixMux) facts.push(["Through", "a Tensix mux"]);
    readout.replaceChildren(
        element("div", { className: "readout-title" }, element("b", {}, `${edge.router.name} → ${downstreamName}`), ` · ${edge.edgeName}`),
        factList(facts));
}

function renderRouterReadout(readout, model, router, pinned) {
    if (!router) {
        readout.replaceChildren();
        return;
    }
    const identity = router.data.identity ?? {};
    const activity = router.data.analysis?.activity ?? {};
    const capture = router.data.analysis?.capture ?? {};
    const stopped = (activity.stopped ?? []).map((side) => `erisc${side.erisc} ${side.side}`).join(", ");
    const regions = capture.regions ? Object.entries(capture.regions).map(([name, count]) => `${count} ${name}`).join(", ") : "";
    const facts = [
        ["Ethernet channel", String(identity.eth_chan ?? "unknown")],
        ["Cores", `logical ${(identity.logical_core ?? []).join(", ")} · virtual ${(identity.virtual_core ?? []).join(", ")}`],
        ["Peer", router.data.link?.peer ?? "none"],
        ["Verdict", `${(activity.verdict ?? "unknown").replace("_", " ")}${stopped ? `: ${stopped} stopped` : ""}`],
        ["Capture", `${capture.status ?? "unknown"}${regions ? ` (regions: ${regions})` : ""}`],
    ];
    const live = Object.entries(router.data.live_downstreams ?? {}).filter(([, directions]) => directions.length)
        .map(([vc, directions]) => `${vc}: ${directions.join(", ")}`);
    facts.push(["Live directions", router.data.live_downstreams === null ? "its one downstream (1D)" : live.join("; ") || "none"]);
    if (router.chip.data.local_sync?.master === router.path) facts.push(["Local sync", "master for this chip"]);
    readout.replaceChildren(
        element("div", { className: "readout-title" }, element("b", {}, router.path),
            ` · ${DIRECTION_NAMES[router.direction] ?? "?"}, plane ${router.plane}`,
            pinned ? element("span", { className: "pill" }, "selected") : null),
        factList(facts));
}

function factList(facts) {
    return element("dl", {}, facts.map(([label, value]) => element("div", {}, element("dt", {}, label), element("dd", {}, value))));
}
