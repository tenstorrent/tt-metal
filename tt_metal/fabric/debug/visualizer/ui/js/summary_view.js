// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The summary strip above the panels: what was captured, on what, how much of it was read, and what decode concluded.

import { element, formatCount, formatDuration, formatUtcTime } from "./format.js";

export function renderSummary(container, model) {
    const facts = element("dl", { className: "facts" }, describeFacts(model).map(([label, value, detail]) =>
        element("div", {},
            element("dt", {}, label),
            element("dd", {}, value),
            detail ? element("dd", { className: "fact-detail" }, detail) : null)));
    const notes = renderNotes(model);
    container.replaceChildren(...(notes ? [facts, notes] : [facts]));
}

function describeFacts(model) {
    const { capture, fabric } = model;
    const startedAt = new Date(capture.started_at);
    const finishedAt = new Date(capture.finished_at);
    const rounds = capture.rounds;
    const routers = [...model.routersByPath.values()];
    return [
        ["Capture", capture.capture_id, capture.cluster_id],
        ["Captured", formatUtcTime(startedAt),
            `took ${formatDuration(finishedAt - startedAt)}` +
            (rounds ? `, ${formatCount(rounds.num_rounds, "round")} ${rounds.interval_ms} ms apart` : "")],
        ["Ranks", `${model.ranks.length} of ${model.worldSize} loaded`, describeHosts(model)],
        ["Build", describeBuildCommit(model), describeBuild(model)],
        ["Fabric", fabric.config?.fabric_config ?? "unknown", `topology ${fabric.topology}`],
        ["Packet header", fabric.packet_header,
            `${formatCount(fabric.packet_header_size_bytes, "byte")} header in ` +
            `${formatCount(fabric.channel_buffer_size_bytes, "byte")} slots`],
        ["Hardware", describeMeshes(model),
            `${formatCount(model.chipsByPath.size, "chip")}, ${formatCount(routers.length, "router")}, ` +
            formatCount(model.links.length, "link")],
        ["Coverage", describeRouterCoverage(model.coverage), describeRegionCoverage(model.coverage)],
        ["Router activity", describeActivity(model, routers), null],
        ["Findings", describeFindings(model), null],
    ];
}

function describeHosts(model) {
    const hostNames = [...new Set(model.ranks.map((rank) => rank.decoded.run.host_name))];
    return hostNames.join(", ");
}

function describeBuildCommit(model) {
    const build = model.ranks[0].decoded.run.build ?? {};
    return `${build.git_sha_short ?? "unknown commit"}${build.git_dirty ? ", dirty" : ""}`;
}

function describeBuild(model) {
    const build = model.ranks[0].decoded.run.build ?? {};
    const parts = [build.version, build.build_type];
    if (model.decode?.git_sha && build.git_sha && model.decode.git_sha !== build.git_sha) {
        parts.push(`decoded by ${model.decode.git_sha.slice(0, 10)}`);
    }
    return parts.filter(Boolean).join(" · ");
}

function describeMeshes(model) {
    return [...model.meshes.values()].map((mesh) => {
        const wrappedAxes = Object.entries(mesh.torus ?? {}).filter(([, wraps]) => wraps).map(([axis]) => axis).sort();
        return `${mesh.id} ${mesh.rows}×${mesh.columns}` + (wrappedAxes.length ? `, wraps ${wrappedAxes.join(" and ")}` : "");
    }).join("; ");
}

function describeRouterCoverage(coverage) {
    return `${coverage.num_routers_complete ?? 0} of ${formatCount(coverage.num_routers ?? 0, "router")} complete`;
}

function describeRegionCoverage(coverage) {
    const parts = [];
    if (coverage.num_routers_partial) parts.push(`${coverage.num_routers_partial} partial`);
    if (coverage.num_routers_not_read) parts.push(`${coverage.num_routers_not_read} not read`);
    parts.push(`${coverage.num_regions_unread ?? 0} of ${formatCount(coverage.num_regions ?? 0, "region")} unread`);
    return parts.join(", ");
}

// Counts each verdict, in the order decode's vocabulary lists them.
function describeActivity(model, routers) {
    const counts = countBy(routers, (router) => router.data.analysis?.activity?.verdict ?? "unknown");
    const order = model.analysisVocabulary?.router_activity ?? [...counts.keys()];
    return order.filter((verdict) => counts.has(verdict)).map((verdict) => `${counts.get(verdict)} ${verdict}`).join(", ") || "none";
}

function describeFindings(model) {
    if (model.findings.length === 0) return "none";
    const counts = countBy(model.findings, (finding) => finding.severity);
    const order = model.analysisVocabulary?.finding_severity ?? [...counts.keys()];
    return order.filter((severity) => counts.has(severity)).map((severity) => `${counts.get(severity)} ${severity}`).join(", ");
}

function countBy(items, keyOf) {
    const counts = new Map();
    for (const item of items) counts.set(keyOf(item), (counts.get(keyOf(item)) ?? 0) + 1);
    return counts;
}

// Warnings from loading and from building the model, and cores the capture skipped. Nothing shows when all is well.
function renderNotes(model) {
    const notes = [...model.loadWarnings, ...model.problems];
    const linksByCheck = countBy(model.links, (link) => link.check);
    if (linksByCheck.get("ends_disagree")) {
        notes.push(`${formatCount(linksByCheck.get("ends_disagree"), "link")} where the two routers don't name each other as peers.`);
    }
    if (linksByCheck.get("peer_not_loaded")) {
        notes.push(`${formatCount(linksByCheck.get("peer_not_loaded"), "link")} to routers on ranks that aren't loaded.`);
    }
    for (const skipped of model.discoverySkippedCores) {
        const core = `ASIC ${skipped.asic_id} core (${skipped.core.x}, ${skipped.core.y})`;
        const where = skipped.router
            ? `${skipped.router.path} (${core}) was still read: ${skipped.router.data.analysis?.capture?.status ?? "unknown"}, ` +
              `verdict ${skipped.router.data.analysis?.activity?.verdict ?? "unknown"}`
            : `${core} isn't a router in the capture`;
        notes.push(`UMD's discovery skipped a core whose Ethernet heartbeat didn't advance. ${where}. UMD said: ${skipped.reason}`);
    }
    if (notes.length === 0) return null;
    return element("ul", { className: "notes" }, notes.map((note) => element("li", {}, note)));
}
