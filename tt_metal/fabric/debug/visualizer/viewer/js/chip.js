import { endpointKey } from "./model.js";
import { resolveRawFile } from "./load.js";

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

function commandFields(root, command, nocSendType) {
  if (!command) {
    return;
  }
  if (command.noc_address) {
    addField(root, "NOC target", nocAddress(command.noc_address));
  }
  if (command.semaphore_noc_address) {
    addField(root, "Semaphore", nocAddress(command.semaphore_noc_address));
  }
  // `value` means different things per send type: the 32-bit payload for an
  // inline write, the semaphore bump for atomic-incs. Label it accordingly.
  const raw = nocSendType?.raw;
  const valueLabel =
    raw === 1
      ? "inline write value"
      : raw === 2 || raw === 3 || raw === 6
        ? "semaphore increment"
        : "value";
  for (const key of ["value", "flush", "chunk_count", "chunk_encoding", "num_dests", "num_chips"]) {
    if (key in command) {
      const label =
        key === "value" ? valueLabel : key === "flush" ? "flush before increment" : key.replaceAll("_", " ");
      addField(root, label, String(command[key]));
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

function hexDump(buffer) {
  const bytes = new Uint8Array(buffer);
  const lines = [];
  for (let offset = 0; offset < bytes.length; offset += 16) {
    const chunk = bytes.slice(offset, offset + 16);
    const hex = [...chunk].map((byte) => byte.toString(16).padStart(2, "0")).join(" ");
    const ascii = [...chunk].map((byte) => (byte >= 32 && byte < 127 ? String.fromCharCode(byte) : ".")).join("");
    lines.push(`${offset.toString(16).padStart(4, "0")}  ${hex.padEnd(47)}  ${ascii}`);
  }
  return lines.join("\n");
}

// Whole-sidecar cache: first raw-byte request loads the sidecar once,
// verifies it against decoded raw_files[], and serves every later slice
// from memory. Keyed by expected identity so two sessions sharing a name
// but not bytes can never cross-contaminate.
const sidecarCache = new Map();

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function rawManifestEntries(raw, refFile) {
  const table = raw?.rawManifest;
  if (!table) {
    return null;
  }
  const base = String(refFile).split("/").at(-1);
  const out = [];
  for (const [key, entries] of table) {
    if (key === refFile || key.split("/").at(-1) === base) {
      out.push(...entries);
    }
  }
  return [...new Map(out.map((entry) => [`${entry.size}|${entry.sha256}`, entry])).values()];
}

async function loadVerifiedSidecar(raw, refFile) {
  const entries = rawManifestEntries(raw, refFile);
  if (entries && entries.length === 0) {
    throw new Error(
      `Sidecar '${refFile}' is not listed in decoded raw_files[] — wrong folder?`,
    );
  }
  if (entries && new Set(entries.map((entry) => entry.sha256)).size > 1) {
    throw new Error(
      `Sidecar '${refFile}' matches ${entries.length} raw_files entries with different bytes — ` +
        "re-gather ranks with unique sidecar names and re-decode",
    );
  }
  const expected = entries?.[0] || null;
  const cacheKey = expected ? `${refFile}|${expected.size}|${expected.sha256}` : `unverified|${refFile}`;
  if (!sidecarCache.has(cacheKey)) {
    sidecarCache.set(
      cacheKey,
      (async () => {
        const local = resolveRawFile(raw?.rawFiles, refFile);
        let bytes;
        let label;
        if (local) {
          if (expected && local.size !== expected.size) {
            throw new Error(
              `Sidecar '${local.name}' has ${local.size} B, decoded expects ${expected.size} B — wrong file gathered?`,
            );
          }
          bytes = await local.arrayBuffer();
          label = local.name;
        } else if (raw?.rawBaseUrl) {
          const url = `${raw.rawBaseUrl}${refFile}`;
          let response;
          try {
            response = await fetch(url, { cache: "no-store" });
          } catch (error) {
            throw new Error(`Could not fetch ${url}: ${error.message}`);
          }
          if (!response.ok) {
            throw new Error(`Could not fetch ${url}: HTTP ${response.status}`);
          }
          bytes = await response.arrayBuffer();
          label = refFile;
        } else {
          throw new Error("Open the decode output folder (JSON + .bin) to read raw bytes");
        }
        if (expected) {
          if (bytes.byteLength !== expected.size) {
            throw new Error(
              `Sidecar '${label}' has ${bytes.byteLength} B, decoded expects ${expected.size} B — wrong file gathered?`,
            );
          }
          // crypto.subtle needs a secure context (localhost/file/https). Without
          // it we keep the size check and say so; with it, bytes are fully pinned.
          if (typeof crypto !== "undefined" && crypto.subtle) {
            const actual = await sha256Hex(bytes);
            if (actual !== expected.sha256) {
              throw new Error(
                `Sidecar '${label}' sha256 ${actual.slice(0, 16)}… does not match ` +
                  `decoded ${expected.sha256.slice(0, 16)}… — wrong rank's file gathered?`,
              );
            }
          }
        }
        return bytes;
      })().catch((error) => {
        sidecarCache.delete(cacheKey);
        throw error;
      }),
    );
  }
  return sidecarCache.get(cacheKey);
}

async function resolveRawBytes(raw, ref) {
  if (!ref) {
    throw new Error("No byte range recorded for this slot");
  }
  const bytes = await loadVerifiedSidecar(raw, ref.file);
  if (ref.offset + ref.size > bytes.byteLength) {
    throw new Error(
      `Range [${ref.offset}, ${ref.offset + ref.size}) exceeds sidecar (${bytes.byteLength} B)`,
    );
  }
  return bytes.slice(ref.offset, ref.offset + ref.size);
}

function rawDropdown(label, ref, raw) {
  const details = el("details", "raw-dropdown");
  const summary = el("summary", null, ref ? `${label} (${ref.size} B)` : `${label} (none)`);
  details.append(summary);
  const body = el("div", "raw-body");
  if (!ref) {
    body.append(el("p", "slot-empty", "No bytes recorded for this range in the decoded file."));
    details.append(body);
    return details;
  }
  body.append(el("p", "slot-empty", "Opening…"));
  details.append(body);
  let loaded = false;
  details.addEventListener("toggle", async () => {
    if (!details.open || loaded) {
      return;
    }
    loaded = true;
    try {
      const buffer = await resolveRawBytes(raw, ref);
      const actions = el("div", "raw-actions");
      const copy = el("button", "raw-copy", "Copy hex");
      copy.type = "button";
      copy.addEventListener("click", async () => {
        const hex = [...new Uint8Array(buffer)]
          .map((byte) => byte.toString(16).padStart(2, "0"))
          .join("");
        try {
          await navigator.clipboard.writeText(hex);
          copy.textContent = "Copied";
        } catch {
          copy.textContent = "Copy failed";
        }
      });
      actions.append(copy);
      const dump = el("pre", "raw-dump", hexDump(buffer));
      body.replaceChildren(actions, dump);
    } catch (error) {
      loaded = false;
      body.replaceChildren(el("p", "slot-error", error instanceof Error ? error.message : String(error)));
    }
  });
  return details;
}

function renderHeaderDetail(root, slot, packetHeaderType, raw = null) {
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
  if (slot.header) {
    const header = slot.header;
    const fields = el("div", "packet-fields");
    addField(fields, "Header type", header.type || packetHeaderType);
    addField(fields, "Payload", `${dash(header.payload_size_bytes)} bytes`);
    addField(fields, "NOC operation", valueName(header.noc_send_type));
    addField(fields, "Source channel", header.src_channel);
    const routing = header.routing || {};
    const destination = routing.destination;
    if (destination) {
      addField(fields, "Destination", `mesh ${destination.mesh_id} · chip ${destination.chip_id}`);
    }
    if (Array.isArray(routing.hops)) {
      const hopLabel = { forward: "FWD", write: "WRITE", write_and_forward: "FWD+WRITE", noop: "—" };
      addField(
        fields,
        "Routing",
        routing.kind === "empty"
          ? "none"
          : `${routing.kind} · ${routing.hops_remaining} hop${routing.hops_remaining === 1 ? "" : "s"} remaining`,
      );
      if (routing.hops.length) {
        addField(
          fields,
          "Hops",
          routing.hops.map((hop, index) => `${index} ${hopLabel[hop] || hop}`).join(" · "),
        );
      }
    }
    if (routing.mcast && Object.values(routing.mcast).some((hops) => hops)) {
      addField(
        fields,
        "Multicast hops",
        ["E", "W", "N", "S"].map((direction) => `${direction} ${dash(routing.mcast[direction])}`).join(" · "),
      );
    }
  if (routing.path) {
    const hops = routing.path.hops || [];
    const line = hops.map((hop, index) => `hop ${index} ${hop}`).join(" · ");
    addField(
      fields,
      "Packet route",
      !hops.length ? "none" : routing.path.complete ? line : `${line} (partial)`,
    );
  }
    commandFields(fields, header.command, header.noc_send_type);
    root.append(fields);
  } else {
    root.append(el("p", "slot-error", slot.error || "The header could not be decoded."));
  }

  const rawSection = el("div", "raw-dropdowns");
  rawSection.append(
    rawDropdown("View raw header bytes", slot.raw_ref, raw),
    rawDropdown("View raw payload bytes", slot.payload_ref, raw),
  );
  root.append(rawSection);
}

export function ringOccupancy(ring, noun = "occupied") {
  if (typeof ring.occupied_count !== "number" || typeof ring.depth !== "number") {
    return `${noun} unknown`;
  }
  return `${ring.occupied_count}/${ring.depth} ${noun}`;
}

export function renderRing(ring, packetHeaderType, subtitle = null, meta = null, stateLine = null, raw = null) {
  const card = el("section", "buffer-card");
  const heading = el("div", "buffer-heading");
  const title = el("div");
  title.append(
    el("h4", null, subtitle ? `${ring.id} (${subtitle})` : ring.id),
    el("span", "buffer-meta", meta ?? `${ringOccupancy(ring)} · ${ring.stride} B/slot`),
  );
  heading.append(title);
  const ringStatus = ring.occupancy_status || ring.status;
  if (ringStatus && ringStatus !== "ok") {
    heading.append(el("span", `status ${ringStatus}`, ringStatus));
  }
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
        renderHeaderDetail(detail, slot, packetHeaderType, raw);
      });
      track.append(button);
    }
  }
  card.append(track);
  if (Array.isArray(stateLine) && stateLine.length) {
    const state = el("div", "buffer-state");
    for (const [name, value] of stateLine) {
      const pair = el("span", "bs-pair");
      pair.append(el("span", "bs-name", `${name}:`), " ", el("span", "bs-value", value));
      state.append(pair);
    }
    card.append(state);
  }
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
    el("span", null, `asic chip id: ${dash(drawnChip.chip.asic_id)}`),
  );
  cardinal.append(center);
  const auxiliary = ["Z", "C", "NONE"].flatMap((direction) => byDirection.get(direction) || []);
  if (auxiliary.length) {
    cardinal.append(rail("aux", auxiliary, selectedKey, onSelect));
  }
  inspector.append(cardinal);

  root.replaceChildren(inspector);
}
