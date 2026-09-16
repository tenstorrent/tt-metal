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

function idTail(id) {
  const parts = String(id).split(".");
  return parts[parts.length - 1] || id;
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

function headerDest(header) {
  const dest = header?.routing?.destination;
  if (!dest) {
    return "";
  }
  return `mesh ${dest.mesh_id} chip ${dest.chip_id}`;
}

function renderSummary(router) {
  const section = el("section", "panel-block");
  section.append(el("h3", null, "Summary"));
  const meta = el("div", "kv-grid");
  const identity = router.identity || {};
  const lifecycle = router.lifecycle || {};
  const liveness = router.liveness || {};
  meta.append(
    kv("endpoint", endpointLabel(router.id)),
    kv("direction", dash(router.direction)),
    kv("link class", dash(router.link_class)),
    kv("plane", dash(router.routing_plane)),
    kv("status", dash(router.capture?.status)),
    kv("identity", identity.matches === true ? "matches" : identity.matches === false ? "mismatch" : "—"),
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
    kv("stall", router.stall_score === null || router.stall_score === undefined ? "—" : Number(router.stall_score).toFixed(2)),
  );
  section.append(meta);
  for (const warning of router.warnings || []) {
    section.append(el("p", "panel-warning", warning));
  }

  const channels = router.channels || { senders: [], receivers: [], downstream: [] };
  section.append(el("h4", null, "Senders"));
  if (!channels.senders?.length) {
    section.append(empty("No enabled senders."));
  } else {
    const table = el("table", "data-table");
    const head = el("tr");
    for (const label of ["idx", "vc", "role", "occupied", "free", "credits", "conn"]) {
      head.append(el("th", null, label));
    }
    table.append(head);
    for (const sender of channels.senders) {
      const row = el("tr");
      row.append(el("td", "mono", sender.index));
      row.append(el("td", "mono", dash(sender.vc)));
      row.append(el("td", null, dash(sender.role)));
      const occ = el("td");
      occ.append(occupancyBar(sender.occupied, sender.depth, sender.status));
      row.append(occ);
      row.append(el("td", "mono", dash(sender.free_slots)));
      const credits = sender.credit_backing === "counter"
        ? `ack ${dash(sender.counters?.to_sender_ack)} / done ${dash(sender.counters?.to_sender_completion)}`
        : `ack ${dash(sender.acked_pending)} / done ${dash(sender.completed_pending)}`;
      row.append(el("td", "mono", credits));
      row.append(el("td", null, dash(sender.connection?.name)));
      table.append(row);
    }
    section.append(table);
  }

  section.append(el("h4", null, "Receivers"));
  if (!channels.receivers?.length) {
    section.append(empty("No enabled receivers."));
  } else {
    const table = el("table", "data-table");
    const head = el("tr");
    for (const label of ["idx", "vc", "pending"]) {
      head.append(el("th", null, label));
    }
    table.append(head);
    for (const receiver of channels.receivers) {
      const row = el("tr");
      row.append(el("td", "mono", receiver.index));
      row.append(el("td", "mono", dash(receiver.vc)));
      const occ = el("td");
      occ.append(occupancyBar(receiver.pkts_pending, receiver.depth, receiver.status));
      row.append(occ);
      table.append(row);
    }
    section.append(table);
  }

  section.append(el("h4", null, "Downstream"));
  if (!channels.downstream?.length) {
    section.append(empty("No enabled downstream edges."));
  } else {
    const table = el("table", "data-table");
    const head = el("tr");
    for (const label of ["vc", "edge", "free", "depth"]) {
      head.append(el("th", null, label));
    }
    table.append(head);
    for (const edge of channels.downstream) {
      const row = el("tr");
      row.append(el("td", "mono", edge.vc));
      row.append(el("td", "mono", edge.edge));
      row.append(el("td", "mono", dash(edge.free_slots)));
      row.append(el("td", "mono", "—"));
      table.append(row);
    }
    section.append(table);
  }
  return section;
}

function renderRings(router) {
  const section = el("section", "panel-block");
  section.append(el("h3", null, "Rings"));
  const rings = router.rings || [];
  if (!rings.length) {
    section.append(empty("No packet rings."));
    return section;
  }
  for (const ring of rings) {
    const details = el("details", "ring-row");
    const summary = el("summary");
    const title = el("span", "ring-id", ring.id);
    const bar = occupancyBar(ring.occupied_count, ring.depth, ring.occupancy_status);
    const status = el("span", `status ${ring.occupancy_status}`, ring.occupancy_status);
    summary.append(title, bar, status);
    details.append(summary);
    const slots = ring.slots || [];
    if (!slots.length) {
      details.append(el("p", "muted", "No slot headers in this file (`--slots none` or ring unread)."));
    } else {
      const table = el("table", "data-table");
      const head = el("tr");
      for (const label of ["slot", "state", "plausible", "dest"]) {
        head.append(el("th", null, label));
      }
      table.append(head);
      for (const slot of slots) {
        const row = el("tr");
        if (slot.header && slot.header.plausible === false) {
          row.classList.add("implausible");
        }
        row.append(el("td", "mono", slot.index));
        row.append(el("td", "mono", slot.slot_state || "unknown"));
        row.append(el("td", "mono", slot.header ? String(Boolean(slot.header.plausible)) : "—"));
        row.append(el("td", "mono", headerDest(slot.header) || dash(slot.error)));
        table.append(row);
      }
      details.append(table);
    }
    section.append(details);
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
    summary.append(el("span", null, region.id || "(root)"), el("span", `status ${region.status || "ok"}`, region.status || "group"));
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
  line.append(el("span", "region-name", idTail(region.id)));
  line.append(el("span", `status ${region.status || "unknown"}`, region.status || "unknown"));
  leaf.append(line);
  if (region.status === "unallocated") {
    return leaf;
  }
  if (region.value !== null && region.value !== undefined) {
    const value = el("div", "region-value");
    value.textContent = formatScalar(region.value);
    if (region.value && typeof region.value === "object") {
      value.title = enumTitle(region.value) || formatScalar(region.value);
    }
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

function renderRegions(router, expert) {
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

export function renderPanel(root, router, { expert = false } = {}) {
  if (!router) {
    root.replaceChildren(empty("Select a router on the map or in the list."));
    return;
  }
  root.replaceChildren(
    renderSummary(router),
    renderRings(router),
    renderRegions(router, expert),
  );
}
