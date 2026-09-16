import { endpointKey } from "./model.js";

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

function valueName(value) {
  if (value && typeof value === "object" && "name" in value) {
    return value.name || `unknown (${dash(value.raw)})`;
  }
  return dash(value);
}

function addField(root, label, value, className = "") {
  const row = el("div", `packet-field${className ? ` ${className}` : ""}`);
  row.append(el("span", "packet-field-name", label), el("span", "packet-field-value", dash(value)));
  root.append(row);
}

function nocAddress(value) {
  if (!value) {
    return null;
  }
  return `x ${dash(value.x)} · y ${dash(value.y)} · local 0x${Number(value.local_addr ?? 0).toString(16)}`;
}

function commandFields(root, command) {
  if (!command) {
    return;
  }
  if (command.noc_address) {
    addField(root, "NOC target", nocAddress(command.noc_address));
  }
  if (command.semaphore_noc_address) {
    addField(root, "Semaphore", nocAddress(command.semaphore_noc_address));
  }
  for (const key of ["value", "flush", "chunk_count", "chunk_encoding", "num_dests", "num_chips"]) {
    if (key in command) {
      addField(root, key.replaceAll("_", " "), String(command[key]));
    }
  }
  if (Array.isArray(command.noc_addresses)) {
    command.noc_addresses.forEach((address, index) => {
      addField(root, `NOC target ${index}`, nocAddress(address));
    });
  }
  if (Array.isArray(command.chunk_sizes)) {
    addField(root, "Chunk sizes", command.chunk_sizes.join(", "));
  }
  if (Array.isArray(command.counts)) {
    addField(root, "Counts", command.counts.join(", "));
  }
}

function renderHeaderDetail(root, slot, packetHeaderType) {
  root.replaceChildren();
  const heading = el("div", "packet-detail-heading");
  heading.append(
    el("div", "eyebrow", `Physical slot ${slot.index}`),
    el("h4", null, slot.header?.plausible ? "Decoded header candidate" : "Residual / implausible header"),
  );
  root.append(heading);

  const warning = el(
    "p",
    `slot-caveat${slot.header?.plausible ? "" : " warning"}`,
    slot.header?.plausible
      ? "This header is structurally plausible, but the capture cannot prove this physical slot is currently occupied."
      : "These bytes do not form a plausible current packet. Ring memory retains old or unused contents.",
  );
  root.append(warning);
  if (!slot.header) {
    root.append(el("p", "slot-error", slot.error || "The header could not be decoded."));
    return;
  }

  const header = slot.header;
  const fields = el("div", "packet-fields");
  addField(fields, "Header type", header.type || packetHeaderType);
  addField(fields, "Payload", `${dash(header.payload_size_bytes)} bytes`);
  addField(fields, "NOC operation", valueName(header.noc_send_type));
  addField(fields, "Source channel", header.src_channel);
  const destination = header.routing?.destination;
  if (destination) {
    addField(fields, "Destination", `mesh ${destination.mesh_id} · chip ${destination.chip_id}`);
  }
  if (header.routing && "value" in header.routing) {
    addField(fields, "Routing value", `0x${Number(header.routing.value).toString(16)}`);
  }
  if (Array.isArray(header.routing?.mcast_params)) {
    addField(fields, "Multicast params", header.routing.mcast_params.join(", "));
  }
  commandFields(fields, header.command);
  root.append(fields);

  if (header.routing?.route_buffer !== undefined) {
    const route = el("details", "route-buffer");
    route.append(el("summary", null, "Route buffer"));
    route.append(el("code", null, Array.isArray(header.routing.route_buffer)
      ? header.routing.route_buffer.join(" ")
      : header.routing.route_buffer));
    root.append(route);
  }
}

export function ringOccupancy(ring) {
  if (typeof ring.occupied_count !== "number" || typeof ring.depth !== "number") {
    return "occupancy unknown";
  }
  return `${ring.occupied_count}/${ring.depth} occupied`;
}

export function renderRing(ring, packetHeaderType) {
  const card = el("section", "buffer-card");
  const heading = el("div", "buffer-heading");
  const title = el("div");
  title.append(el("h4", null, ring.id), el("span", "buffer-meta", `${ringOccupancy(ring)} · stride ${ring.stride} B`));
  const status = el("span", `status ${ring.occupancy_status || ring.status}`, ring.occupancy_status || ring.status);
  heading.append(title, status);
  card.append(heading);

  const track = el("div", "slot-track");
  const detail = el("div", "packet-detail");
  const slots = ring.slots || [];
  if (!slots.length) {
    track.append(el("p", "slot-empty", "No slot headers in this file. Decode without `--slots none` to inspect them."));
  } else {
    for (const slot of slots) {
      const plausible = slot.header?.plausible === true;
      const button = el("button", `slot-cell${plausible ? " plausible" : ""}`);
      button.type = "button";
      button.title = plausible
        ? `Slot ${slot.index}: plausible residual header`
        : `Slot ${slot.index}: unknown state, implausible or undecoded header`;
      button.append(el("span", "slot-index", slot.index), el("span", "slot-mark", plausible ? "H" : "·"));
      button.addEventListener("click", () => {
        for (const sibling of track.querySelectorAll(".slot-cell.selected")) {
          sibling.classList.remove("selected");
        }
        button.classList.add("selected");
        renderHeaderDetail(detail, slot, packetHeaderType);
      });
      track.append(button);
    }
  }
  card.append(track);
  card.append(el(
    "p",
    "buffer-caveat",
    "Cells are physical ring slots. H marks a plausible decoded header, not a known occupied slot.",
  ));
  if (slots.length) {
    detail.append(el("p", "slot-empty", "Select a slot to inspect its decoded packet header."));
  }
  card.append(detail);
  return card;
}

function directionOrder(direction) {
  return ["N", "E", "S", "W", "Z", "C", "NONE"].indexOf(direction);
}

function routerButton(router, selectedKey, onSelect) {
  const key = endpointKey(router.id);
  const button = el("button", `chip-router${key === selectedKey ? " selected" : ""}`);
  button.type = "button";
  button.dataset.direction = router.direction || "NONE";
  const main = el("span", "chip-router-main");
  main.append(el("b", null, router.direction || "—"), el("span", null, `eth ${router.id.eth_chan}`));
  const meta = el("span", "chip-router-meta");
  meta.textContent = [
    router.routing_plane === null || router.routing_plane === undefined ? null : `plane ${router.routing_plane}`,
    router.capture?.status,
    router.stall_score === null || router.stall_score === undefined ? null : `stall ${Number(router.stall_score).toFixed(2)}`,
  ].filter(Boolean).join(" · ");
  button.append(main, meta);
  button.addEventListener("click", () => onSelect?.(key));
  return button;
}

function rail(direction, routers, selectedKey, onSelect) {
  const root = el("div", `router-rail rail-${direction.toLowerCase()}`);
  root.dataset.direction = direction;
  for (const router of routers) {
    root.append(routerButton(router, selectedKey, onSelect));
  }
  return root;
}

function emptyRail(direction) {
  const root = el("div", `router-rail rail-${direction.toLowerCase()} empty`);
  root.dataset.direction = direction;
  root.append(el("span", null, direction));
  return root;
}

export function renderCardinal(root, model, layout, chipKey, selectedKey, onSelect) {
  const drawnChip = layout.chips.find((chip) => chip.key === chipKey);
  if (!drawnChip) {
    root.replaceChildren(el("p", "slot-empty", `Chip ${chipKey} is unavailable.`));
    return;
  }
  const routers = (drawnChip.chip.routers || [])
    .map((endpoint) => model.byEndpoint.get(endpointKey(endpoint)))
    .filter(Boolean)
    .sort((left, right) =>
      directionOrder(left.direction) - directionOrder(right.direction)
      || (left.routing_plane ?? 0) - (right.routing_plane ?? 0)
      || left.id.eth_chan - right.id.eth_chan);
  const byDirection = new Map();
  for (const router of routers) {
    const direction = router.direction || "NONE";
    if (!byDirection.has(direction)) {
      byDirection.set(direction, []);
    }
    byDirection.get(direction).push(router);
  }

  const inspector = el("div", "chip-inspector");
  const intro = el("div", "chip-intro");
  intro.append(
    el("div", "eyebrow", "Chip detail"),
    el("h3", null, `Mesh ${drawnChip.meshId} · chip ${drawnChip.chipId}`),
    el("p", "muted", `${routers.length} fabric routers · select a cardinal port`),
  );
  inspector.append(intro);

  const cardinal = el("section", "chip-cardinal");
  for (const direction of ["N", "E", "S", "W"]) {
    cardinal.append(byDirection.has(direction)
      ? rail(direction, byDirection.get(direction), selectedKey, onSelect)
      : emptyRail(direction));
  }
  const center = el("div", "chip-center");
  center.append(
    el("span", "chip-center-coord", `coord [${(drawnChip.chip.mesh_coord || ["?", "?"]).join(", ")}]`),
    el("strong", null, `${drawnChip.meshId}:${drawnChip.chipId}`),
    el("span", null, dash(drawnChip.chip.asic_id)),
  );
  cardinal.append(center);
  const auxiliary = ["Z", "C", "NONE"].flatMap((direction) => byDirection.get(direction) || []);
  if (auxiliary.length) {
    cardinal.append(rail("aux", auxiliary, selectedKey, onSelect));
  }
  inspector.append(cardinal);

  root.replaceChildren(inspector);
}
