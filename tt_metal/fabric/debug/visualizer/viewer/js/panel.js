import { endpointLabel } from "./model.js";

function el(name, className, text) {
  const node = document.createElement(name);
  if (className) {
    node.className = className;
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  return node;
}

function dash(value) {
  return value === null || value === undefined || value === "" ? "—" : value;
}

function enumLabel(value) {
  if (!value || typeof value !== "object") {
    return dash(value);
  }
  if (value.name) {
    return value.name;
  }
  if (value.raw === null || value.raw === undefined) {
    return "—";
  }
  return String(value.raw);
}

function enumTitle(value) {
  if (!value || typeof value !== "object") {
    return "";
  }
  const parts = [];
  if (value.name) {
    parts.push(value.name);
  }
  if (value.raw !== null && value.raw !== undefined) {
    parts.push(`raw ${value.raw}`);
  }
  return parts.join(" · ");
}

// Siblings often share a tail (`credits.downstream.vc0.edge3.free_slots` vs
// `credits.vc2_receiver.free_slots`), so label a region by what its parent does not say.
function relativeId(region) {
  const id = String(region.id ?? "");
  const parent = String(region.parent ?? "");
  if (parent && id.startsWith(`${parent}.`)) {
    return id.slice(parent.length + 1);
  }
  return id;
}

function hasKey(value, key) {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function hex32(raw) {
  return typeof raw === "number" ? `0x${(raw >>> 0).toString(16).padStart(8, "0")}` : String(raw);
}

function chipList(entries) {
  const list = el("div", "value-chips");
  for (const [label, value] of entries) {
    const chip = el("span", "value-chip");
    chip.append(el("b", null, label), el("span", null, value));
    list.append(chip);
  }
  return list;
}

function jsonDetails(value) {
  const details = el("details", "region-json");
  details.append(el("summary", null, `${Object.keys(value).length} fields`));
  details.append(el("pre", null, JSON.stringify(value, null, 2)));
  return details;
}

// Formats by decoded value shape, not by builder names: counter arrays are one u32 per
// channel, stream registers carry pre/post, u32 regions carry a word plus its neighbors.
function regionValueNode(region) {
  const value = region.value;
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    return el("div", "region-value", formatScalar(value));
  }
  if (hasKey(value, "name") || hasKey(value, "raw")) {
    const node = el("div", "region-value", enumLabel(value));
    node.title = enumTitle(value);
    return node;
  }
  if (Array.isArray(value.counters)) {
    const stride = region.stride ? ` · ${region.stride} B stride` : "";
    const node = chipList(value.counters.map((counter, index) => [`ch${index}`, String(counter)]));
    node.title = `one counter per sender channel${stride}`;
    return node;
  }
  if (hasKey(value, "pre") && hasKey(value, "post")) {
    const node = el(
      "div",
      "region-value",
      value.torn ? `${dash(value.pre)} → ${dash(value.post)} (moved during capture)` : dash(value.post),
    );
    node.title = `pre ${dash(value.pre)} · post ${dash(value.post)}`;
    return node;
  }
  if (Array.isArray(value.words)) {
    const trailing = value.words.slice(1);
    const text = trailing.every((word) => word === 0) ? dash(value.word) : value.words.join(" ");
    return el("div", "region-value", text);
  }
  if (Array.isArray(value.samples)) {
    const node = el("div", "region-value", value.samples.map((sample) => hex32(sample.raw)).join(" → "));
    node.title = value.samples.map((sample) => `${sample.t}: ${sample.raw}`).join("\n");
    return node;
  }
  const entries = Object.entries(value);
  if (entries.some(([, item]) => item !== null && typeof item === "object")) {
    return jsonDetails(value);
  }
  return chipList(entries.map(([key, item]) => [key, formatScalar(item)]));
}

function formatScalar(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "object") {
    if (Object.prototype.hasOwnProperty.call(value, "name") || Object.prototype.hasOwnProperty.call(value, "raw")) {
      return enumLabel(value);
    }
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

function kv(label, value, title) {
  const row = el("div", "kv");
  row.append(el("span", "k", label), el("span", "v", value));
  if (title) {
    row.title = title;
  }
  return row;
}

function empty(text) {
  return el("p", "empty-list", text);
}

function statusBadge(status, fallback = "unknown") {
  const name = status || fallback;
  if (!name || name === "ok") {
    return null;
  }
  return el("span", `status ${name}`, name);
}

export function renderSummary(router) {
  const section = el("section", "router-card");
  const lifecycle = router.lifecycle || {};
  const liveness = router.liveness || {};

  const heading = el("div", "router-card-heading");
  heading.append(el("div", "eyebrow", "Selected router"));
  const title = el("h3", null, `${router.direction || "—"} · eth ${router.id.eth_chan} · plane ${dash(router.routing_plane)}`);
  title.title = endpointLabel(router.id);
  heading.append(title);
  heading.append(el("p", "muted", `${endpointLabel(router.id)} · ${dash(router.link_class)}`));
  section.append(heading);

  const badges = el("div", "value-chips");
  const status = statusBadge(router.capture?.status);
  if (status) {
    badges.append(status);
  }
  section.append(badges);

  const meta = el("div", "kv-grid lifecycle-grid");
  const sampleCount = Array.isArray(liveness.samples) ? liveness.samples.length : 0;
  meta.append(
    kv("EDM status:", enumLabel(lifecycle.edm_status), enumTitle(lifecycle.edm_status)),
    kv(
      "Termination signal:",
      enumLabel(lifecycle.termination_signal),
      enumTitle(lifecycle.termination_signal),
    ),
    kv("Host run message:", enumLabel(lifecycle.go_signal), enumTitle(lifecycle.go_signal)),
    kv(
      `Heartbeat (sampled ${sampleCount} times):`,
      liveness.classification
        ? `${liveness.classification} (fabric ${liveness.fabric_samples}, base_fw ${liveness.base_fw_samples})`
        : "—",
    ),
  );
  section.append(meta);
  for (const warning of router.warnings || []) {
    section.append(el("p", "panel-warning", warning));
  }
  return section;
}

export function senderCredits(sender) {
  if (sender.credit_backing === "counter") {
    return `ack ${dash(sender.counters?.to_sender_ack)} / done ${dash(sender.counters?.to_sender_completion)}`;
  }
  return `ack ${dash(sender.acked_pending)} / done ${dash(sender.completed_pending)}`;
}

export function senderRoleLabel(sender) {
  if (sender.producer === "worker") {
    return "local worker";
  }
  if (sender.producer) {
    return `from ${sender.producer}`;
  }
  return dash(sender.role);
}

const DIRECTION_NAMES = { E: "East", W: "West", N: "North", S: "South", Z: "Z" };

function siblingEgress(model, router, direction) {
  const sameChip = (model.routers || []).filter(
    (candidate) =>
      candidate.id?.mesh_id === router.id?.mesh_id &&
      candidate.id?.chip_id === router.id?.chip_id &&
      candidate.direction === direction,
  );
  if (!sameChip.length) {
    return null;
  }
  return sameChip.find((candidate) => candidate.routing_plane === router.routing_plane) ?? sameChip[0];
}

function destSenderStatus(model, router, edge) {
  const sibling = edge.direction ? siblingEgress(model, router, edge.direction) : null;
  const slot = typeof edge.dest_sender_channel === "number" ? edge.dest_sender_channel : null;
  const dest =
    sibling === null || slot === null
      ? null
      : (sibling.channels?.senders || []).find((sender) => sender.index === slot) ?? null;
  const name = edge.direction ? `${DIRECTION_NAMES[edge.direction] || edge.direction} Edge` : `Edge ${edge.edge}`;
  const channel = slot === null ? "—" : String(slot);
  const status =
    sibling === null
      ? "egress router not captured"
      : dest !== null && typeof dest.occupied === "number" && typeof dest.depth === "number"
        ? `${dest.occupied}/${dest.depth} occupied slots`
        : "not captured";
  return `${name} (Sender Channel ${channel}) Status: ${status}`;
}

export function renderDownstreamEdges(model, router, vc = null) {
  const edges = [...(router.channels?.downstream || [])]
    .filter((edge) => vc === null || edge.vc === vc)
    .sort((left, right) => (left.vc === right.vc ? left.edge - right.edge : left.vc - right.vc));
  if (!edges.length) {
    return null;
  }
  const card = el("section", "buffer-card");
  const heading = el("div", "buffer-heading");
  const title = el("div");
  title.append(el("h4", null, "downstream edges (sender channels) for receiver to forward to"));
  heading.append(title);
  card.append(heading);
  for (const edge of edges) {
    card.append(el("span", "buffer-meta", destSenderStatus(model, router, edge)));
  }
  return card;
}

function regionStatusBadges(region) {
  const badges = [];
  const unallocated = region.status === "unallocated";
  if (region.enabled === false && !unallocated) {
    badges.push(el("span", "status disabled", "disabled"));
  }
  const status = statusBadge(region.status, region.backing === "group" ? "group" : "unknown");
  if (status) {
    badges.push(status);
  }
  return badges;
}

function renderRegionNode(region, children, byParent, expert) {
  const isGroup = region.backing === "group";
  const disabled = region.enabled === false;
  if (isGroup) {
    const folder = el("details", `region-group${disabled ? " muted" : ""}`);
    folder.open = region.parent === "";
    const summary = el("summary");
    const label = el("span", "region-label");
    const name = el("span", null, relativeId(region) || "(root)");
    name.title = region.id || "";
    label.append(name, ...regionStatusBadges(region));
    summary.append(label);
    folder.append(summary);
    const nested = el("div", "region-children");
    for (const child of children) {
      nested.append(renderRegionNode(child, byParent.get(child.id) || [], byParent, expert));
    }
    folder.append(nested);
    return folder;
  }

  const leaf = el("div", `region-leaf${disabled ? " muted" : ""}`);
  const line = el("div", "region-line");
  const label = el("span", "region-label");
  const name = el("span", "region-name", relativeId(region));
  name.title = region.id || "";
  label.append(name, ...regionStatusBadges(region));
  line.append(label);
  leaf.append(line);
  if (region.status === "unallocated") {
    return leaf;
  }
  const value = regionValueNode(region);
  if (value) {
    leaf.append(value);
  }
  if (expert && region.raw_hex) {
    leaf.append(el("pre", "raw-hex", region.raw_hex));
  }
  for (const child of children) {
    leaf.append(renderRegionNode(child, byParent.get(child.id) || [], byParent, expert));
  }
  return leaf;
}

export function renderRegions(router, expert) {
  const section = el("section", "panel-block");
  section.append(el("h3", null, "Regions"));
  const regions = router.regions || [];
  if (!regions.length) {
    section.append(empty("No regions."));
    return section;
  }
  const byParent = new Map();
  for (const region of regions) {
    const parent = region.parent || "";
    if (!byParent.has(parent)) {
      byParent.set(parent, []);
    }
    byParent.get(parent).push(region);
  }
  const hasHex = regions.some((region) => Boolean(region.raw_hex));
  if (expert && !hasHex) {
    section.append(el("p", "muted", "This file was decoded without --expert-raw; no hex is embedded."));
  }
  const roots = byParent.get("") || regions.filter((region) => !region.parent);
  const tree = el("div", "region-tree");
  for (const region of roots) {
    tree.append(renderRegionNode(region, byParent.get(region.id) || [], byParent, expert));
  }
  section.append(tree);
  return section;
}

export function emptyNotice(text) {
  return empty(text);
}
