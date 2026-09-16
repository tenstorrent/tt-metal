import { renderRing, ringOccupancy } from "./chip.js";
import {
  edgeDirection,
  emptyNotice,
  renderChannels,
  renderRegions,
  renderSummary,
  senderCredits,
  senderRoleLabel,
} from "./panel.js";

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

function regionWord(region) {
  const word = region?.value?.word;
  return typeof word === "number" ? String(word) : "—";
}

function senderStateLine(sender, cursorRegion, flowRegion, termRegion) {
  const cursor = cursorRegion?.value || {};
  const writeIndex = typeof cursor.write_index === "number" ? cursor.write_index : "—";
  const writeCounter = typeof cursor.write_counter === "number" ? cursor.write_counter : "—";
  return [
    ["connection", sender.connection?.name ?? "—"],
    ["write_idx", `${writeIndex} (ctr ${writeCounter})`],
    ["flow semaphore", regionWord(flowRegion)],
    ["termination status", regionWord(termRegion)],
  ];
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
    const senders = new Map((router.channels?.senders || []).map((sender) => [sender.index, sender]));
    const receivers = new Map((router.channels?.receivers || []).map((receiver) => [receiver.index, receiver]));
    const regions = new Map((router.regions || []).map((region) => [region.id, region]));
    const edges = router.channels?.downstream || [];
    for (const ring of router.rings) {
      const senderMatch = /^sender\.(\d+)\.ring$/.exec(ring.id || "");
      const receiverMatch = /^receiver\.(\d+)\.ring$/.exec(ring.id || "");
      const slot = ring.stride === null || ring.stride === undefined ? null : `${ring.stride} B/slot`;
      let subtitle = null;
      let meta = null;
      let stateLine = null;
      if (senderMatch && senders.has(Number(senderMatch[1]))) {
        const flat = Number(senderMatch[1]);
        const sender = senders.get(flat);
        subtitle = senderRoleLabel(sender);
        meta = [ringOccupancy(ring), senderCredits(sender), slot].filter(Boolean).join(" · ");
        stateLine = senderStateLine(sender, regions.get(`sender.${flat}.control.buffer_index_sem`), regions.get(`sender.${flat}.control.flow_semaphore`), regions.get(`sender.${flat}.control.termination_status`));
      } else if (receiverMatch && receivers.has(Number(receiverMatch[1]))) {
        const receiver = receivers.get(Number(receiverMatch[1]));
        const hops = edges
          .filter((edge) => edge.vc === receiver.vc)
          .map((edge) => {
            const direction = edgeDirection(router.direction, edge.edge);
            const free = edge.free_slots === null || edge.free_slots === undefined ? "—" : edge.free_slots;
            return `${direction ?? `e${edge.edge}`} ${free}`;
          });
        meta = [ringOccupancy(ring, "pending"), hops.length ? `next ${hops.join(" · ")}` : null, slot]
          .filter(Boolean)
          .join(" · ");
      }
      wrap.append(renderRing(ring, packetHeaderType, subtitle, meta, stateLine));
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
