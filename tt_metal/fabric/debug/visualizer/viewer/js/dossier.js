import { renderRing } from "./chip.js";
import { emptyNotice, renderChannels, renderRegions, renderSummary } from "./panel.js";

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

function renderBuffers(router, packetHeaderType) {
  const wrap = el("div", "dossier-buffers");
  const heading = el("div", "buffers-title");
  heading.append(
    el("div", "eyebrow", "Buffer inspector"),
    el("h3", null, `${router.direction} · eth ${router.id.eth_chan} · plane ${dash(router.routing_plane)}`),
    el("p", "muted", packetHeaderType),
  );
  wrap.append(heading);
  if (!router.rings?.length) {
    wrap.append(el("p", "slot-empty", "No packet rings are described for this router."));
  } else {
    for (const ring of router.rings) {
      wrap.append(renderRing(ring, packetHeaderType));
    }
  }
  return wrap;
}

export function renderDossier(root, model, router, { expert = false } = {}) {
  if (!router) {
    root.replaceChildren(emptyNotice("Select a router on the map, on the chip, or in the list."));
    return;
  }
  root.replaceChildren(
    renderBuffers(router, model.decoded.fabric_context.packet_header_type),
    renderSummary(router),
    renderChannels(router),
    renderRegions(router, expert),
  );
}
