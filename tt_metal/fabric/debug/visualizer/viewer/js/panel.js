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

function occupancyBar(used, depth, status) {
  const wrap = el("div", "occ-bar");
  if (status === "inconsistent") {
    wrap.classList.add("inconsistent");
    wrap.append(el("span", "occ-warn", `${used}/${depth} inconsistent`));
    return wrap;
  }
  if (typeof used !== "number" || typeof depth !== "number" || depth <= 0) {
    wrap.append(el("span", "occ-label", `${dash(used)}/${dash(depth)}`));
    return wrap;
  }
  const fill = el("span", "occ-fill");
  fill.style.width = `${Math.min(100, Math.max(0, (used / depth) * 100))}%`;
  wrap.append(fill, el("span", "occ-label", `${used}/${depth}`));
  return wrap;
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

export function renderSummary(router) {
  const section = el("section", "router-card");
  const identity = router.identity || {};
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
  const status = el("span", `status ${router.capture?.status || "unknown"}`, router.capture?.status || "unknown");
  badges.append(status);
  const identityBadge = el(
    "span",
    `status ${identity.matches === true ? "ok" : identity.matches === false ? "reset" : "unknown"}`,
    identity.matches === true ? "identity matches" : identity.matches === false ? "identity mismatch" : "identity —",
  );
  badges.append(identityBadge);
  badges.append(
    el(
      "span",
      "status",
      router.stall_score === null || router.stall_score === undefined
        ? "stall —"
        : `stall ${Number(router.stall_score).toFixed(2)}`,
    ),
  );
  section.append(badges);

  const meta = el("div", "kv-grid");
  meta.append(
    kv("exit", dash(lifecycle.exit_state), enumTitle(lifecycle.edm_status)),
    kv("EDM", enumLabel(lifecycle.edm_status), enumTitle(lifecycle.edm_status)),
    kv("term", enumLabel(lifecycle.termination_signal), enumTitle(lifecycle.termination_signal)),
    kv("go", enumLabel(lifecycle.go_signal), enumTitle(lifecycle.go_signal)),
    kv(
      "liveness",
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

function senderCredits(sender) {
  if (sender.credit_backing === "counter") {
    return `ack ${dash(sender.counters?.to_sender_ack)} / done ${dash(sender.counters?.to_sender_completion)}`;
  }
  return `ack ${dash(sender.acked_pending)} / done ${dash(sender.completed_pending)}`;
}

function renderSenderRow(sender) {
  const row = el("div", `sender-row${sender.status && sender.status !== "ok" ? ` status-${sender.status}` : ""}`);
  const id = el("span", "mono");
  id.append(el("b", null, `ch${sender.index}`));
  row.append(id);
  row.append(el("span", "role-chip", dash(sender.role)));
  row.append(el("span", "muted mono", sender.vc === null || sender.vc === undefined ? "—" : `VC${sender.vc}`));
  const occ = el("span", "sender-occ");
  occ.append(occupancyBar(sender.occupied, sender.depth, sender.status));
  row.append(occ);
  row.append(el("span", "mono", `free ${dash(sender.free_slots)}`));
  const credits = el("span", "mono muted", senderCredits(sender));
  credits.title =
    sender.credit_backing === "counter"
      ? "backed by on-device counters"
      : "backed by credit streams";
  row.append(credits);
  row.append(el("span", "muted", dash(sender.connection?.name)));
  return row;
}

function renderReceiverChip(receiver) {
  const chip = el(
    "span",
    `value-chip${receiver.status && receiver.status !== "ok" ? ` status-${receiver.status}` : ""}`,
  );
  const pending =
    typeof receiver.pkts_pending === "number" && typeof receiver.depth === "number"
      ? `${receiver.pkts_pending}/${receiver.depth}`
      : `${dash(receiver.pkts_pending)}/${dash(receiver.depth)}`;
  chip.append(el("b", null, `ch${receiver.index}`), el("span", null, `VC${dash(receiver.vc)} · pending ${pending}`));
  chip.title = `receiver ${receiver.index}, status ${receiver.status || "unknown"}`;
  return chip;
}

function renderEdgeChip(edge) {
  const free = edge.free_slots;
  const state = free === 0 ? "starved" : free === null || free === undefined ? "unknown" : "open";
  const chip = el("span", `value-chip edge-chip edge-${state}`);
  chip.append(el("b", null, `vc${edge.vc}:e${edge.edge}`), el("span", null, `free ${dash(free)}`));
  if (state === "starved") {
    chip.title = "downstream edge reports zero free slots — backpressure candidate";
  }
  return chip;
}

export function renderChannels(router) {
  const section = el("section", "channels-card");
  const heading = el("div", "buffers-title");
  heading.append(
    el("div", "eyebrow", "Channels"),
    el("h3", null, "Senders · receivers · downstream"),
    el("p", "muted", "Occupancy from streams; credit returns as ack/done pairs."),
  );
  section.append(heading);

  const channels = router.channels || { senders: [], receivers: [], downstream: [] };
  section.append(el("h4", null, "Senders"));
  if (!channels.senders?.length) {
    section.append(empty("No enabled senders."));
  } else {
    const list = el("div", "sender-list");
    for (const sender of channels.senders) {
      list.append(renderSenderRow(sender));
    }
    section.append(list);
  }

  section.append(el("h4", null, "Receivers"));
  if (!channels.receivers?.length) {
    section.append(empty("No enabled receivers."));
  } else {
    const list = el("div", "value-chips");
    for (const receiver of channels.receivers) {
      list.append(renderReceiverChip(receiver));
    }
    section.append(list);
  }

  section.append(el("h4", null, "Downstream edges"));
  if (!channels.downstream?.length) {
    section.append(empty("No enabled downstream edges."));
  } else {
    const list = el("div", "value-chips");
    for (const edge of channels.downstream) {
      list.append(renderEdgeChip(edge));
    }
    section.append(list);
  }
  return section;
}

function renderRegionNode(region, children, byParent, expert) {
  const isGroup = region.backing === "group";
  const disabled = region.enabled === false;
  if (isGroup) {
    const folder = el("details", `region-group${disabled ? " muted" : ""}`);
    folder.open = region.parent === "";
    const summary = el("summary");
    const name = el("span", null, relativeId(region) || "(root)");
    name.title = region.id || "";
    summary.append(name);
    if (disabled) {
      summary.append(el("span", "status disabled", "disabled"));
    }
    summary.append(el("span", `status ${region.status || "ok"}`, region.status || "group"));
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
  const name = el("span", "region-name", relativeId(region));
  name.title = region.id || "";
  line.append(name);
  const badges = el("span", "region-badges");
  if (disabled && region.status !== "unallocated") {
    badges.append(el("span", "status disabled", "disabled"));
  }
  badges.append(el("span", `status ${region.status || "unknown"}`, region.status || "unknown"));
  line.append(badges);
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
