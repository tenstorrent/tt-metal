// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The router view: one router's facts and activity, then its channels VC by VC. Each sender and receiver card shows
// its ring buffer's slots and what capture read about them: free slots or packets held, packets not credited back
// yet, the credit counters the channel's credits come through, and the connection. The capture gives counts, not
// which slots hold a packet, so every slot is drawn the same and the counts are shown beside them.

import { DIRECTION_NAMES } from "./capture_model.js";
import { element, formatCount, formatDuration, formatHex } from "./format.js";
import { VERDICT_DESCRIPTIONS } from "./hardware_view.js";
import { renderFieldTable, renderFieldsByCategory, renderPlainValue } from "./field_view.js";
import { renderHeaderSummary, renderSlotHeader } from "./packet_view.js";
import { renderRoutingTable } from "./routing_table_view.js";

const CHANNEL_STATUS_NOTES = {
    vc_not_serviced: "This router doesn't service the VC, so the channel has no slots.",
    mux: "The router's tensix mux carries this channel, so it has no slots in the router.",
    trimmed: "The channel trimming profile turned this channel off, so it has no slots.",
};
const TAKEN_AT_NOTES = {
    retire: "The receiver takes a packet once it has issued the packet's forward or local write.",
    first_level_ack: "The receiver takes a packet when it acks it with first-level ack.",
};
// The peer's credit counter array that pairs with each of this router's: the to_sender arrays count credits the
// router has received for its own sender channels; the receiver arrays count credits it has sent for the peer's.
const PAIRED_ARRAYS = {
    to_sender_ack: "receiver_ack",
    to_sender_completion: "receiver_completion",
    receiver_ack: "to_sender_ack",
    receiver_completion: "to_sender_completion",
};

// Builds the view in `panel`. onSelectRouter(routerPath | null) opens another router (a peer, producer or edge), or
// closes the view with null. onShowPath({ label, routerPath, kind, path }) draws a slot's packet path on the hardware
// view.
export function createRouterView(panel, model, { onSelectRouter, onShowPath }) {
    panel.hidden = true;

    function showRouter(routerPath) {
        const router = routerPath ? model.routersByPath.get(routerPath) ?? null : null;
        panel.hidden = !router;
        if (!router) {
            panel.replaceChildren();
            return;
        }
        const closeButton = element("button", { type: "button", className: "ghost" }, "Close");
        closeButton.addEventListener("click", () => onSelectRouter(null));
        const swatch = element("span", { className: "swatch" });
        swatch.style.background = router.plane === null ? "var(--fg-muted)" : `var(--plane-${router.plane % 4})`;
        panel.replaceChildren(
            element("div", { className: "panel-head" },
                element("h2", { className: "panel-title" }, swatch, ` Router ${router.path}`), closeButton),
            renderFacts(model, router, onSelectRouter),
            renderEriscs(router),
            renderActivity(router),
            renderPacketLayout(model.fabric),
            ...renderVcs(model, router, { onSelectRouter, onShowPath }),
            renderCreditCounters(model, router),
            element("section", { className: "router-section" },
                element("h3", { className: "section-title" }, "Router fields"),
                element("p", { className: "hint" }, "Every field the manifest records for the router, by ERISC and category. Channel fields are in each channel's card."),
                ...renderFieldsByCategory(router.data.fields, model.vocabulary?.field_category)),
            renderRoutingTable(model, router, onSelectRouter),
        );
    }

    return { showRouter };
}

// ---------- Facts and activity ----------

function routerButton(path, onSelectRouter, text = path) {
    const button = element("button", { type: "button", className: "link-button" }, text);
    button.addEventListener("click", () => onSelectRouter(path));
    return button;
}

function renderFacts(model, router, onSelectRouter) {
    const identity = router.data.identity ?? {};
    const link = router.data.link ?? {};
    const capture = router.data.analysis?.capture ?? {};
    const peerPath = link.peer;
    const linkNotes = [link.edge_capability?.replaceAll("_", " ")];
    if (link.wrap) linkNotes.push("wrap");
    if (link.cross_host) linkNotes.push("crosses MPI ranks");
    if (link.dispatch_link) linkNotes.push("dispatch link");
    const regions = Object.entries(capture.regions ?? {}).map(([name, count]) => `${count} ${name}`).join(", ");
    const facts = [
        ["Direction", `${DIRECTION_NAMES[router.direction] ?? "unknown"}, plane ${router.plane ?? "?"}`],
        ["Ethernet channel", String(identity.eth_chan ?? "unknown")],
        ["Cores", `logical ${(identity.logical_core ?? []).join(", ")} · virtual ${(identity.virtual_core ?? []).join(", ")}`],
        ["Peer", peerPath ? (model.routersByPath.has(peerPath) ? routerButton(peerPath, onSelectRouter) : `${peerPath} (not loaded)`) : "none"],
        ["Link", linkNotes.filter(Boolean).join(", ") || "none"],
        ["Capture", `${(capture.status ?? "unknown").replace("_", " ")}${regions ? ` (${regions} regions)` : ""}`],
    ];
    if (router.chip.data.local_sync?.master === router.path) facts.push(["Local sync", "master for its chip"]);
    return element("dl", { className: "facts" }, facts.map(([label, value]) => element("div", {}, element("dt", {}, label), element("dd", {}, value))));
}

// One subsection per ERISC the router's kernel runs on, each listing what the decoded file records for it under the
// file's own key names (processor, physical_erisc, opt_level, and any key added later).
function renderEriscs(router) {
    const eriscs = Object.entries(router.data.eriscs ?? {});
    const activeCount = router.data.shape?.num_active_eriscs;
    return element("section", { className: "router-section" },
        element("h3", { className: "section-title" }, "ERISCs"),
        element("p", { className: "hint" }, `The ERISCs the router's kernel runs on, in RISC order${activeCount !== undefined ? ` (num_active_eriscs: ${activeCount})` : ""}.`),
        eriscs.length ? element("div", { className: "erisc-cards" }, eriscs.map(([name, erisc]) => element("div", { className: "card erisc-card" },
            element("div", { className: "card-head" }, element("b", {}, name)),
            element("dl", { className: "card-facts" }, Object.entries(erisc ?? {}).map(([key, value]) =>
                element("div", {}, element("dt", {}, key), element("dd", {}, renderPlainValue(value)))))))) :
            element("p", { className: "hint" }, "None recorded."));
}

// The verdict, and every heartbeat read it came from: one row per heartbeat, one column per round.
function renderActivity(router) {
    const activity = router.data.analysis?.activity ?? {};
    const verdict = activity.verdict ?? "unknown";
    const heartbeats = [];
    if (activity.loop_heartbeat) heartbeats.push({ label: "Loop heartbeat", reads: activity.loop_heartbeat.reads ?? [], value: (read) => read.count, moved: activity.loop_heartbeat.moved, written: true });
    for (const heartbeat of activity.telemetry_heartbeats ?? []) {
        heartbeats.push({ label: `erisc${heartbeat.erisc} ${heartbeat.side} telemetry`, reads: heartbeat.reads ?? [], value: (read) => read.value, moved: heartbeat.moved, written: heartbeat.written });
    }
    const rounds = [...new Set(heartbeats.flatMap((heartbeat) => heartbeat.reads.map((read) => read.round)))]
        .sort((first, second) => (first === null) - (second === null) || first - second);
    const stopped = new Set((activity.stopped ?? []).map((side) => `erisc${side.erisc} ${side.side} telemetry`));
    const table = element("table", { className: "data-table" },
        element("thead", {}, element("tr", {}, element("th", { scope: "col" }, "Heartbeat"),
            rounds.map((round) => element("th", { scope: "col" }, round === null ? "Final read" : `Round ${round}`)),
            element("th", { scope: "col" }, "Moved"))),
        element("tbody", {}, heartbeats.map((heartbeat) => element("tr", { className: stopped.has(heartbeat.label) ? "stopped" : "" },
            element("th", { scope: "row" }, heartbeat.label),
            heartbeat.written ? rounds.map((round) => {
                const read = heartbeat.reads.find((candidate) => candidate.round === round);
                return element("td", {}, read ? String(heartbeat.value(read)) : "—");
            }) : element("td", { colSpan: rounds.length || 1, className: "faint" }, "never written"),
            element("td", {}, heartbeat.written ? (heartbeat.moved ? "yes" : "no") : "—")))));
    const stoppedText = (activity.stopped ?? []).map((side) => `erisc${side.erisc} ${side.side}`).join(", ");
    return element("section", { className: "router-section" },
        element("h3", { className: "section-title" }, "Activity"),
        element("p", { className: "verdict-line" }, element("span", { className: `verdict-dot verdict-${verdict}` }),
            element("b", {}, verdict.replace("_", " ")), `: ${VERDICT_DESCRIPTIONS[verdict] ?? ""}`,
            stoppedText ? `. Stopped: ${stoppedText}.` : ""),
        heartbeats.length ? element("div", { className: "table-wrap" }, table) : element("p", { className: "hint" }, "No heartbeat reads."));
}

// A slot is a packet header then its payload: the bar is drawn to scale.
function renderPacketLayout(fabric) {
    const headerBytes = fabric?.packet_header_size_bytes;
    const slotBytes = fabric?.channel_buffer_size_bytes;
    const payloadBytes = fabric?.max_payload_size_bytes;
    if (!headerBytes || !slotBytes) return null;
    const headerShare = Math.max((headerBytes / slotBytes) * 100, 4);
    return element("section", { className: "router-section" },
        element("h3", { className: "section-title" }, "Packet slot"),
        element("div", { className: "slot-anatomy" },
            element("div", { className: "anatomy-bar" },
                element("span", { className: "anatomy-header", style: { flexBasis: `${headerShare}%` } }),
                element("span", { className: "anatomy-payload" })),
            element("p", { className: "anatomy-labels" },
                element("span", { className: "anatomy-key header" }), element("b", {}, `${headerBytes} B`), ` header (${fabric.packet_header})`,
                element("span", { className: "anatomy-key payload" }), element("b", {}, `${(payloadBytes ?? slotBytes - headerBytes).toLocaleString("en-US")} B`),
                " payload at most · ", element("b", {}, `${slotBytes.toLocaleString("en-US")} B`), " per slot")));
}

// ---------- Channels ----------

function renderVcs(model, router, callbacks) {
    const channels = router.data.channels ?? {};
    const shape = router.data.shape ?? {};
    const vcs = [...new Set([...Object.keys(channels.senders ?? {}), ...Object.keys(channels.receivers ?? {})])]
        .sort((first, second) => Number(first.slice(2)) - Number(second.slice(2)));
    const blocks = vcs.map((vc) => {
        const senders = Object.entries(channels.senders?.[vc] ?? {});
        const receivers = Object.entries(channels.receivers?.[vc] ?? {});
        const peerPath = router.data.link?.peer ?? "the peer";
        return element("section", { className: "router-section vc-block" },
            element("div", { className: "vc-head" }, element("h3", {}, vc.toUpperCase()),
                element("span", {}, `${formatCount(senders.length, "sender")}, ${formatCount(receivers.length, "receiver")}`)),
            element("div", { className: "vc-columns" },
                element("div", { className: "vc-column" }, element("p", { className: "column-title" }, `Senders · out to ${peerPath}`),
                    senders.length ? senders.map(([name, channel]) => renderSender(model, router, vc, name, channel, callbacks))
                        : element("p", { className: "hint" }, "None on this VC.")),
                element("div", { className: "vc-column" }, element("p", { className: "column-title" }, `Receivers · in from ${peerPath}`),
                    receivers.length ? receivers.map(([name, channel]) => renderReceiver(model, router, vc, name, channel, callbacks))
                        : element("p", { className: "hint" }, "None on this VC."))));
    });
    if (shape.num_vcs !== undefined && vcs.length < shape.num_vcs) {
        blocks.push(element("p", { className: "hint" }, `The router has ${formatCount(shape.num_vcs, "VC")}; only ${vcs.join(", ")} have channels.`));
    }
    return blocks;
}

function firstField(fieldsByErisc, name) {
    for (const fields of Object.values(fieldsByErisc ?? {})) if (fields?.[name]) return fields[name];
    return null;
}

function channelCard(kind, name, channel, rows, extra) {
    const status = channel.status ?? "unknown";
    const head = element("div", { className: "card-head" }, element("b", {}, `${kind} ${name}`),
        element("span", { className: `pill status-${status}` }, status.replaceAll("_", " ")));
    if (status !== "active") {
        return element("div", { className: "card off" }, head,
            element("p", { className: "hint" }, CHANNEL_STATUS_NOTES[status] ?? `The channel is ${status}.`), ...extra.filter(Boolean));
    }
    return element("div", { className: "card" }, head,
        element("dl", { className: "card-facts" }, rows.filter(Boolean).map(([label, value, note]) => element("div", {},
            element("dt", {}, label), element("dd", {}, value, note ? element("span", { className: "note-line" }, note) : null)))),
        ...extra.filter(Boolean));
}

// A bar filled to count of depth. Every bar shows slots in use (not sent, held, or not credited back), never which.
function meter(count, depth, tone) {
    const share = depth ? Math.min(Math.max(count / depth, 0), 1) * 100 : 0;
    const bar = element("span", { className: `meter ${tone}`, role: "img", ariaLabel: `${count} of ${depth}` }, element("span", { style: { width: `${share}%` } }));
    return bar;
}

function renderSender(model, router, vc, name, channel, callbacks) {
    const { onSelectRouter } = callbacks;
    const freeSlots = firstField(channel.fields, "free_slots")?.decoded ?? null;
    const connection = firstField(channel.fields, "connection")?.decoded?.value ?? null;
    const connectionInfo = firstField(channel.fields, "conn_info")?.decoded?.value ?? null;
    const cursor = firstField(channel.fields, "buffer_index_sem")?.decoded?.value ?? null;
    const injection = firstField(channel.fields, "is_injection_channel");
    const uncredited = channel.analysis?.uncredited_at_peer;

    let fedBy = "unknown";
    if (channel.producer === "worker") {
        const xy = connectionInfo?.worker_xy;
        fedBy = `a worker${xy ? ` at NoC (${xy.x}, ${xy.y})` : ""}`;
    } else if (channel.producer) {
        fedBy = element("span", {}, "sibling ", model.routersByPath.has(channel.producer) ? routerButton(channel.producer, onSelectRouter, channel.producer.split("/").at(-1)) : channel.producer);
    }

    const rows = [["Fed by", fedBy, injection?.value ? "an injection channel" : null],
        ["Serviced by", (channel.serviced_by ?? []).join(", ") || "none"]];
    if (freeSlots && freeSlots.depth !== null && freeSlots.depth !== undefined) {
        const unsent = freeSlots.unsent ?? freeSlots.depth - freeSlots.count;
        rows.push(["Not sent yet", element("span", { className: "meter-line" }, meter(unsent, freeSlots.depth, unsent ? "busy" : "idle"),
            `${unsent} of ${freeSlots.depth} slots`), `free_slots reads ${freeSlots.count}: slots not holding a packet waiting to be sent`]);
    } else if (freeSlots) {
        rows.push(["Free slots", String(freeSlots.count ?? freeSlots.raw)]);
    }
    if (uncredited) {
        rows.push(["Uncredited at peer", element("span", { className: "meter-line" },
            meter(uncredited.value, freeSlots?.depth ?? 0, uncredited.value ? "waiting" : "idle"), `${uncredited.value} packets`),
        `sent over Ethernet, credits not back to ${channel.producer?.split("/").at(-1) ?? "the producer"} yet: its edge's ${uncredited.producer_uncredited} uncredited less ${uncredited.unsent} not sent here, read ${formatDuration(uncredited.read_apart_ns / 1e6)} apart`]);
    } else if (channel.producer === "worker") {
        rows.push(["Uncredited at peer", element("span", { className: "faint" }, "not known"), "a worker has no edge count to compare against"]);
    }
    for (const [erisc, credits] of Object.entries(channel.credits ?? {})) {
        for (const [creditName, reference] of Object.entries(credits ?? {})) {
            const { label, value, note } = describeCredit(model, router, creditName, reference);
            rows.push([label, value, Object.keys(channel.credits).length > 1 ? `${erisc}: ${note}` : note]);
        }
    }
    if (connection || connectionInfo) {
        const producerName = channel.producer === "worker" ? "worker" : channel.producer?.split("/").at(-1) ?? "producer";
        rows.push([`Producer connection`, keyValueList([
            ["connection", connection ? renderPlainValue(connection) : null],
            ["worker_xy", connectionInfo?.worker_xy ? renderPlainValue(connectionInfo.worker_xy) : null],
            ["edm_read_counter", connectionInfo?.edm_read_counter],
        ]), `the ${producerName}'s connection to this channel: connection is the state the producer and router write, conn_info where the producer is and the router's read counter`]);
    }
    if (cursor) {
        rows.push(["Producer cursor", keyValueList(Object.entries(cursor)),
            "buffer_index_sem: where the worker keeps its write cursor across connections. It doesn't say which slots hold packets."]);
    }
    return channelCard("Sender", name, channel, rows,
        [renderSlotStrip(model, router, { kind: "sender", vc, name }, channel.ring_buffer, callbacks), renderChannelFields(channel.fields)]);
}

// Keys and values on their own lines, under the decoded file's key names. Null values are left out.
function keyValueList(entries) {
    return element("dl", { className: "key-values" }, entries.filter(([, value]) => value !== null && value !== undefined)
        .map(([key, value]) => element("div", {}, element("dt", {}, key), element("dd", {}, typeof value === "object" ? value : String(value)))));
}

// A sender channel's credit counter. `completed` is a running total since startup of the channel's packets the peer's
// receiver has finished with (their forwards or writes done); `acked`, of those it has acked with first-level ack. The
// peer keeps the same total in its receiver array and copies it here over Ethernet; for each one the router hasn't
// processed yet (a count in a register, not captured), it frees a slot in the peer's receiver and returns a credit to
// the producer. Shown beside the peer's own total.
const CREDIT_LABELS = {
    completed: ["Completions from peer", "packets from this channel the peer's receiver has finished with, in total since startup"],
    acked: ["Acks from peer", "packets from this channel the peer's receiver has acked with first-level ack, in total since startup"],
};

function describeCredit(model, router, creditName, reference) {
    const arrayName = reference?.array?.split("/").at(-1);
    const index = reference?.index;
    const own = readCounter(router, arrayName, index);
    const peer = model.routersByPath.get(router.data.link?.peer ?? "");
    const pairedName = PAIRED_ARRAYS[arrayName];
    const paired = peer && pairedName ? readCounter(peer, pairedName, index) : null;
    const [label, meaning] = CREDIT_LABELS[creditName] ?? [`Credits ${creditName}`, "a credit counter"];
    return {
        label,
        value: keyValueList([
            [`${arrayName}[${index}]`, element("b", {}, String(own ?? "?"))],
            peer ? [`${peer.name}'s ${pairedName}[${index}]`, element("b", {}, String(paired ?? "?"))] : [null, null],
        ]),
        note: `${meaning}. The first is this router's copy, the second the peer's own count.`,
    };
}

function readCounter(router, arrayName, index) {
    const value = router.data.credit_counters?.[arrayName]?.decoded?.elements?.[index];
    return value === undefined ? null : value;
}

function renderReceiver(model, router, vc, name, channel, callbacks) {
    const { onSelectRouter } = callbacks;
    const packetsSent = firstField(channel.fields, "pkts_sent")?.decoded ?? null;
    const groupSize = firstField(channel.fields, "completion_group_size")?.value;
    const rows = [["Serviced by", (channel.serviced_by ?? []).join(", ") || "none"],
        ["Forwards on", [...new Set(Object.values(channel.forwards_on ?? {}))].join(", ") || "none"]];
    if (packetsSent && packetsSent.depth !== null && packetsSent.depth !== undefined) {
        rows.push(["Held", element("span", { className: "meter-line" },
            meter(packetsSent.count, packetsSent.depth, packetsSent.count >= packetsSent.depth ? "full" : packetsSent.count ? "busy" : "idle"),
            `${packetsSent.count} of ${packetsSent.depth} slots`),
        `pkts_sent: packets the peer sent that the receiver hasn't taken yet. ${TAKEN_AT_NOTES[packetsSent.taken_at] ?? ""}`]);
    } else if (packetsSent) {
        rows.push(["Packets sent", String(packetsSent.count ?? packetsSent.raw)]);
    }
    if (groupSize !== undefined) rows.push(["Completion group", `${groupSize} slot${groupSize === 1 ? "" : "s"}`, "credits go back to the peer a group at a time"]);
    rows.push(["Credits sent to peer", renderCreditsToPeer(model, router, vc),
        "per sender channel of the peer: the running total of its packets this receiver has finished with (completed) or acked, beside the peer's copy of the same total. Hover a line for the counters."]);
    return channelCard("Receiver", name, channel, rows,
        [renderDownstreamEdges(model, router, vc, onSelectRouter),
            renderSlotStrip(model, router, { kind: "receiver", vc, name }, channel.ring_buffer, callbacks), renderChannelFields(channel.fields)]);
}

// The credits this router has sent back for each of the peer's sender channels on the VC, beside what the peer has
// received: this router's receiver arrays and the peer's to_sender arrays, both indexed by the peer's channel.
function renderCreditsToPeer(model, router, vc) {
    const peer = model.routersByPath.get(router.data.link?.peer ?? "");
    if (!peer) return element("span", { className: "faint" }, "the peer isn't loaded, so its channels aren't known; see the credit counters below");
    const lines = [];
    for (const [channelName, channel] of Object.entries(peer.data.channels?.senders?.[vc] ?? {})) {
        for (const credits of Object.values(channel.credits ?? {})) {
            for (const [creditName, reference] of Object.entries(credits ?? {})) {
                const peerArray = reference.array?.split("/").at(-1);
                const ownArray = PAIRED_ARRAYS[peerArray];
                const own = readCounter(router, ownArray, reference.index);
                const theirs = readCounter(peer, peerArray, reference.index);
                lines.push(element("span", { title: `${ownArray}[${reference.index}] here, ${peerArray}[${reference.index}] on ${peer.path}` },
                    element("span", { className: "faint" }, `${peer.name} ${channelName} ${creditName}: `),
                    element("b", {}, String(own ?? "?")), " sent",
                    element("span", { className: "faint" }, ` · ${peer.name} received `), element("b", {}, String(theirs ?? "?"))));
            }
        }
    }
    return lines.length ? element("span", { className: "stacked" }, lines) : element("span", { className: "faint" }, "none");
}

// Each sibling sender channel the receiver forwards into, with the free slots the router sees there.
function renderDownstreamEdges(model, router, vc, onSelectRouter) {
    const edges = Object.entries(router.data.intra_chip_downstream_edges?.[vc] ?? {});
    if (!edges.length) return null;
    // In 1D live_downstreams is null: a packet's route only says whether to forward, to the one downstream.
    const live = router.data.live_downstreams === null || router.data.live_downstreams === undefined ? null : router.data.live_downstreams[vc] ?? [];
    const directionWords = { N: "north", E: "east", S: "south", W: "west", Z: "z" };
    return element("div", { className: "edges-block" },
        element("p", { className: "column-title" }, "Forwards into"),
        element("ul", { className: "edge-list" }, edges.map(([edgeName, edge]) => {
            const [downstreamPath, channelPath] = (edge.downstream_channel ?? "").split("/channels/");
            const downstream = model.routersByPath.get(downstreamPath);
            const freeSlots = firstField(edge.fields, "free_slots")?.decoded ?? null;
            const isLive = live === null || (downstream ? live.includes(directionWords[downstream.direction]) : false);
            const target = element("span", {}, downstream ? routerButton(downstreamPath, onSelectRouter, downstream.name) : downstreamPath,
                ` ${(channelPath ?? "").replace("senders/", "").replaceAll("/", " ")}`);
            const counts = freeSlots && freeSlots.depth !== null && freeSlots.depth !== undefined
                ? element("span", { className: "meter-line" }, meter(freeSlots.depth - freeSlots.count, freeSlots.depth, freeSlots.count === 0 ? "full" : freeSlots.count < freeSlots.depth ? "busy" : "idle"),
                    `${freeSlots.count} of ${freeSlots.depth} free`, freeSlots.uncredited ? element("span", { className: "faint" }, ` · ${freeSlots.uncredited} uncredited`) : null)
                : element("span", { className: "faint" }, freeSlots ? `free_slots ${freeSlots.count ?? freeSlots.raw}` : "no free slots read");
            return element("li", { className: isLive ? "live" : "idle" },
                element("span", { className: "edge-name" }, edgeName), target, counts,
                element("span", { className: "faint" }, isLive ? (live === null ? "forwards here (1D)" : "a live direction") : "connected, not a live direction"),
                edge.through_tensix_mux ? element("span", { className: "faint" }, "through the tensix mux") : null);
        })));
}

// One box per slot. Capture has counts, not which slots hold a packet, so every slot with a header is drawn alike; a
// slot never written (its header all zeros) is hollow. Hovering a slot shows its address and the header last written
// to it, which may be from a packet long gone; clicking it opens that header and its path below the slots.
function renderSlotStrip(model, router, slotChannel, ringBuffer, { onSelectRouter, onShowPath }) {
    if (!ringBuffer) return null;
    const slotCount = ringBuffer.num_elements ?? 0;
    const slotBytes = ringBuffer.element_size ?? 0;
    const headers = ringBuffer.decoded?.elements ?? [];
    const neverWritten = headers.filter((header) => header === null).length;
    const readout = element("p", { className: "slot-readout" }, "Hover a slot for its address and last header; click it to open the header.");
    const detail = element("div", { className: "slot-detail-holder" });
    const strip = element("div", { className: "slots", role: "group", ariaLabel: "Slots" });
    strip.style.setProperty("--slot-count", String(slotCount));
    const slotButtons = [];
    let openIndex = null;
    const addressOf = (index) => ringBuffer.address + index * slotBytes;
    const channelLabel = `${router.path} ${slotChannel.kind} ${slotChannel.vc} ${slotChannel.name}`;

    // Opens slot `index`, or closes it when it's already open.
    function openSlot(index) {
        openIndex = openIndex === index ? null : index;
        slotButtons.forEach((button, buttonIndex) => button.setAttribute("aria-pressed", String(buttonIndex === openIndex)));
        if (openIndex === null) {
            detail.replaceChildren();
            return;
        }
        const header = headers[index] ?? null;
        detail.replaceChildren(renderSlotHeader(model, { router, kind: slotChannel.kind, index, address: addressOf(index), header }, {
            onSelectRouter,
            onShowPath: () => onShowPath?.({ label: `${channelLabel} slot ${index}`, routerPath: router.path, kind: slotChannel.kind, path: header?.path ?? [] }),
        }));
    }

    for (let index = 0; index < slotCount; index++) {
        const header = headers[index] ?? null;
        const slot = element("button", { type: "button", className: header ? "slot" : "slot never-written", ariaLabel: `Slot ${index}` });
        slot.setAttribute("aria-pressed", "false");
        const describe = () => {
            readout.textContent = `Slot ${index} of ${slotCount} · ${formatHex(addressOf(index))} · ${describeHeader(header)}`;
        };
        slot.addEventListener("pointerenter", describe);
        slot.addEventListener("focus", describe);
        slot.addEventListener("click", () => openSlot(index));
        slotButtons.push(slot);
        strip.append(slot);
    }
    return element("div", { className: "slot-block" },
        element("p", { className: "column-title" }, "Slots"), strip,
        element("p", { className: "slots-caption" },
            `${slotCount} slots × ${slotBytes.toLocaleString("en-US")} B = ${(slotCount * slotBytes).toLocaleString("en-US")} B at ${formatHex(ringBuffer.address)}`,
            neverWritten ? ` · ${neverWritten} never written (hollow)` : ""),
        readout,
        detail,
        renderHeaderSummary(model, headers, onSelectRouter));
}

function describeHeader(header) {
    if (!header) return "never written";
    const sendType = header.noc_send_type?.name ?? header.noc_send_type?.value ?? "unknown send type";
    const destination = header.route?.dst ?? header.path?.at(-1)?.chip ?? null;
    return `last header: ${sendType}, ${(header.payload_size_bytes ?? 0).toLocaleString("en-US")} B payload${destination ? `, to ${destination}` : ""}`;
}

function renderChannelFields(fieldsByErisc) {
    const count = new Set(Object.values(fieldsByErisc ?? {}).flatMap((fields) => Object.keys(fields ?? {}))).size;
    if (!count) return null;
    return element("details", { className: "field-group" }, element("summary", {}, "All fields", element("span", { className: "faint" }, ` · ${count}`)),
        renderFieldTable(fieldsByErisc));
}

// ---------- Credit counters ----------

// The router's four credit counter arrays, one row each. A cell a channel's credits point at is named on hover; the
// to_sender arrays are indexed by this router's sender channels, the receiver arrays by the peer's.
function renderCreditCounters(model, router) {
    const arrays = Object.entries(router.data.credit_counters ?? {});
    if (!arrays.length) return null;
    const peer = model.routersByPath.get(router.data.link?.peer ?? "");
    const ownNames = channelNamesByCounter(router);
    const peerNames = peer ? channelNamesByCounter(peer) : new Map();
    const width = Math.max(...arrays.map(([, counter]) => counter.decoded?.elements?.length ?? 0), 0);
    const table = element("table", { className: "data-table counters-table" },
        element("thead", {}, element("tr", {}, element("th", { scope: "col" }, "Array"),
            Array.from({ length: width }, (_, index) => element("th", { scope: "col" }, String(index))))),
        element("tbody", {}, arrays.map(([arrayName, counter]) => {
            const ownIndexed = counter.indexed_by === "own_sender_channel";
            const names = ownIndexed ? ownNames : peerNames;
            const owner = ownIndexed ? router.name : peer?.name ?? "the peer";
            return element("tr", {},
                element("th", { scope: "row", title: `${counter.indexed_by?.replaceAll("_", " ")} · at ${formatHex(counter.address)}` }, arrayName,
                    element("span", { className: "faint" }, ` · ${owner}'s channels`)),
                Array.from({ length: width }, (_, index) => {
                    const value = counter.decoded?.elements?.[index];
                    const channelName = names.get(index);
                    return element("td", { className: channelName ? "used" : "unused", title: channelName ? `${owner} ${channelName}` : "no channel here" },
                        value === undefined ? "" : String(value));
                }));
        })));
    return element("section", { className: "router-section" },
        element("h3", { className: "section-title" }, "Credit counters"),
        element("p", { className: "hint" }, "Sender channels get their credits back through these L1 counters. Columns are channels by compact index; the bright ones belong to a channel (hover for which)."),
        element("div", { className: "table-wrap" }, table));
}

// Which sender channel each counter index belongs to, from the channels' credits: { index → "vc0 ch1" }.
function channelNamesByCounter(router) {
    const names = new Map();
    for (const [vc, channels] of Object.entries(router.data.channels?.senders ?? {})) {
        for (const [channelName, channel] of Object.entries(channels ?? {})) {
            for (const credits of Object.values(channel.credits ?? {})) {
                for (const reference of Object.values(credits ?? {})) names.set(reference.index, `${vc} ${channelName}`);
            }
        }
    }
    return names;
}
