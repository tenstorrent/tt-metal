# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for what channel counts and credits say, on a hand-written manifest and hand-made decoded routers.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_flow_control
"""

import copy
import pathlib
import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.decode.flow_control import Count, ProducerCore, get_channel_flows
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter, ElementValue, StreamValue

DEPTH = 8
WORKER, FED, UNFED, TRIMMED = (f"channels/senders/vc0/ch{index}" for index in range(4))
RECEIVER = "channels/receivers/vc0/ch0"
UNUSED = {"raw": 0, "name": "unused_connection_value"}
OPEN = {"raw": 1, "name": "open_connection_value"}
CLOSE_REQUESTED = {"raw": 2, "name": "close_connection_request_value"}


def stream_ref(stream_id):
    return {"stream_id": stream_id, "register": "buf_space_available", "schema": "u32"}


def credit_ref(backing, index, array="to_sender_completion"):
    return (
        stream_ref(20 + index)
        if backing == "stream_register"
        else {"array": f"credit_counters/{array}", "index": index}
    )


def sender(index, producer, backing, status="active", acked=False):
    credits = {"completed": credit_ref(backing, index)}
    if acked:
        credits["acked"] = credit_ref(backing, index, "to_sender_ack")
    return {
        "status": status,
        "producer": producer,
        "ring_buffer": {"num_elements": DEPTH},
        "credits": credits,
        "fields": {"free_slots": stream_ref(index)},
    }


def counters():
    """The router's four credit counter arrays: its own senders index to_sender_*, its peer's index receiver_*."""
    own, peer = {"index_space": "own_sender_compact"}, {"index_space": "peer_sender_compact"}
    return {"to_sender_ack": own, "to_sender_completion": own, "receiver_ack": peer, "receiver_completion": peer}


def router_node(peer, senders=None, receivers=None, edges=None, first_level_ack=False, speedy=False):
    return {
        "link": {"peer": peer},
        "shape": {"vc0_bubble_flow_control": first_level_ack},
        "fields": {
            "speedy_vc0": {"kind": "flag", "value": speedy},
            "sender_credit_amortization": {"kind": "number", "value": 4 if speedy else 0},
        },
        "credit_counters": counters(),
        "intra_chip_downstream_edges": edges or {},
        "channels": {"senders": {"vc0": senders or {}}, "receivers": {"vc0": receivers or {}}},
    }


def manifest(backing="stream_register", first_level_ack=False, speedy=False):
    """
    C0's east router, E0, whose link reaches C1's west router. Its sender ch0 is the worker channel, ch1 is fed by C0's
    south router, ch2 by nothing, and ch3 is trimmed.
    """
    senders = {
        "ch0": sender(0, "worker", backing, acked=first_level_ack),
        "ch1": sender(1, "M0/C0/S0", backing),
        "ch2": sender(2, None, backing),
        "ch3": sender(3, None, backing, status="trimmed"),
    }
    receivers = {
        "ch0": {"status": "active", "ring_buffer": {"num_elements": DEPTH}, "fields": {"pkts_sent": stream_ref(9)}}
    }
    edge = {"downstream_channel": f"M0/C0/E0/{FED}", "fields": {"free_slots": stream_ref(30)}}
    return {
        "enums": {
            "ConnectionValue": {
                "values": {
                    "unused_connection_value": 0,
                    "open_connection_value": 1,
                    "close_connection_request_value": 2,
                }
            }
        },
        "meshes": {
            "M0": {
                "credit_transport": {
                    "vc0": {"backing": backing, "reasons": [] if backing == "stream_register" else ["multi_txq"]}
                },
                "chips": {
                    "C0": {
                        "routers": {
                            "E0": router_node(
                                "M0/C1/W0", senders, receivers, first_level_ack=first_level_ack, speedy=speedy
                            ),
                            "S0": router_node(None, edges={"vc0": {"edge1": edge}}),
                        }
                    },
                    "C1": {"routers": {"W0": router_node("M0/C0/E0")}},
                },
            }
        },
    }


def stream(before, after=None):
    after = before if after is None else after
    return StreamValue(0, "buf_space_available", 17, before, after, before, after)


def worker_regions(connection=UNUSED, written=0, credited=0, core=(1, 2)):
    return {
        f"{WORKER}/fields/connection": connection,
        f"{WORKER}/fields/conn_info": {
            "worker_semaphore_address": 0,
            "worker_teardown_semaphore_address": 0,
            "worker_xy": {"x": core[0], "y": core[1]},
            "edm_read_counter": credited,
        },
        f"{WORKER}/fields/buffer_index_sem": {"write_counter": written, "write_index": written % DEPTH},
    }


def east_router(streams=None, regions=None, elements=None):
    """E0, idle: every sender's free_slots at the depth, its credits at 0, and nothing waiting in its receiver."""
    all_streams = {f"channels/senders/vc0/ch{index}/fields/free_slots": stream(DEPTH) for index in range(4)}
    all_streams.update({f"channels/senders/vc0/ch{index}/credits/completed": stream(0) for index in range(4)})
    all_streams[f"{RECEIVER}/fields/pkts_sent"] = stream(0)
    all_streams.update(streams or {})
    return DecodedRouter("M0/C0/E0", None, {}, {**worker_regions(), **(regions or {})}, all_streams, elements or {})


def sibling_router(free_slots=DEPTH):
    streams = {"intra_chip_downstream_edges/vc0/edge1/fields/free_slots": stream(free_slots)}
    return DecodedRouter("M0/C0/S0", None, {}, {}, streams, {})


def flows(routers, manifest_=None):
    snapshot = Input(
        pathlib.Path("snapshot.json"),
        "0" * 64,
        {"routers": {}},
        b"",
        pathlib.Path("manifest.json"),
        manifest_ or manifest(),
        (),
    )
    by_path = {router.path: router for router in routers}
    return get_channel_flows(snapshot, by_path, "M0/C0/E0")


class SenderCountsTest(unittest.TestCase):
    def test_an_idle_router_has_nothing_unsent_or_uncredited(self):
        result = flows([east_router(), sibling_router()])
        self.assertEqual(list(result), [WORKER, FED, UNFED, RECEIVER])
        for channel in (WORKER, FED):
            with self.subTest(channel):
                self.assertEqual((result[channel].unsent, result[channel].uncredited), (Count(0, 0), Count(0, 0)))
                self.assertIs(result[channel].within_bounds, True)
                self.assertIs(result[channel].moved, False)

    def test_unsent_is_the_depth_less_free_slots(self):
        result = flows([east_router({f"{FED}/fields/free_slots": stream(5, 6)}), sibling_router(3)])
        self.assertEqual(result[FED].unsent, Count(3, 2))
        self.assertIs(result[FED].moved, True)

    def test_a_sibling_fed_channels_uncredited_comes_from_the_siblings_edge(self):
        # The sibling has forwarded 5 packets the router hasn't credited back; 3 of them are still unsent.
        result = flows([east_router({f"{FED}/fields/free_slots": stream(5)}), sibling_router(3)])
        self.assertEqual((result[FED].unsent, result[FED].uncredited), (Count(3, 3), Count(5, 5)))
        self.assertIs(result[FED].within_bounds, True)

    def test_a_sibling_fed_channel_ignores_buffer_index_sem(self):
        # Nothing writes a sibling-fed channel's cursor, and on Wormhole it holds whatever L1 held.
        junk = {f"{FED}/fields/buffer_index_sem": {"write_counter": 1797810135, "write_index": 3175682171}}
        result = flows([east_router(regions=junk), sibling_router()])
        self.assertEqual(result[FED].uncredited, Count(0, 0))
        self.assertIsNone(result[FED].connection)

    def test_a_missing_sibling_isnt_observable(self):
        self.assertEqual(
            flows([east_router()])[FED].uncredited.not_observable, "the producer, M0/C0/S0, isn't in this snapshot"
        )

    def test_a_channel_nothing_feeds_has_no_uncredited(self):
        self.assertEqual(
            flows([east_router(), sibling_router()])[UNFED].uncredited.not_observable, "nothing feeds the channel"
        )

    def test_unsent_above_uncredited_is_out_of_bounds(self):
        result = flows([east_router({f"{FED}/fields/free_slots": stream(2)}), sibling_router(7)])
        self.assertIs(result[FED].within_bounds, False)


class WorkerChannelTest(unittest.TestCase):
    def test_a_disconnected_worker_channel_counts_from_the_cursor_and_read_counter(self):
        # The last worker wrote 7 packets; the router has credited 4.
        result = flows([east_router(regions=worker_regions(written=7, credited=4)), sibling_router()])
        worker = result[WORKER]
        self.assertEqual(worker.uncredited, Count(3, 3))
        self.assertEqual(worker.connection, UNUSED)
        self.assertEqual(worker.producer_core, ProducerCore(1, 2, "last"))

    def test_the_counters_wrap_at_2_to_the_32(self):
        regions = worker_regions(written=3, credited=0xFFFFFFFE)
        self.assertEqual(flows([east_router(regions=regions), sibling_router()])[WORKER].uncredited, Count(5, 5))

    def test_a_connected_worker_channels_uncredited_isnt_observable(self):
        for connection in (OPEN, CLOSE_REQUESTED):
            with self.subTest(connection["name"]):
                regions = worker_regions(connection, written=7, credited=4)
                worker = flows([east_router(regions=regions), sibling_router()])[WORKER]
                self.assertEqual(worker.connection, connection)
                self.assertEqual(worker.producer_core, ProducerCore(1, 2, "current"))
                self.assertIsNone(worker.uncredited.before)
                self.assertIn("a worker is connected", worker.uncredited.not_observable)

    def test_a_connection_word_that_names_no_state(self):
        regions = worker_regions({"raw": 7, "name": None})
        worker = flows([east_router(regions=regions), sibling_router()])[WORKER]
        self.assertEqual(worker.uncredited.not_observable, "the connection word holds 7, which names no state")


class CreditsTest(unittest.TestCase):
    def test_stream_register_credits_are_pending_only(self):
        result = flows([east_router({f"{WORKER}/credits/completed": stream(2, 1)}), sibling_router()])
        credit = result[WORKER].credits["completed"]
        self.assertEqual(
            (credit.backing, credit.frees, credit.pending), ("stream_register", "receiver_slot", Count(2, 1))
        )
        self.assertIn("used up", credit.total_landed.not_observable)
        self.assertIn("used up", credit.on_link.not_observable)

    def l1_flows(self, landed, peer_totals):
        elements = {
            f"channels/senders/vc0/ch{index}/credits/completed": ElementValue(
                "credit_counters/to_sender_completion", index, landed[index]
            )
            for index in range(4)
        }
        routers = [east_router(elements=elements), sibling_router()]
        if peer_totals is not None:
            routers.append(
                DecodedRouter("M0/C1/W0", None, {}, {"credit_counters/receiver_completion": peer_totals}, {}, {})
            )
        return flows(routers, manifest("l1_counter"))

    def test_l1_counter_credits_are_totals_checked_against_the_peers(self):
        # Channel 1's peer has sent one completion that hasn't landed yet.
        result = self.l1_flows([5, 6, 0, 0], [5, 7, 0, 0])
        worker, fed = result[WORKER].credits["completed"], result[FED].credits["completed"]
        self.assertEqual((worker.backing, worker.backing_reasons), ("l1_counter", ["multi_txq"]))
        self.assertIn("own variables", worker.pending.not_observable)
        self.assertEqual(
            (worker.total_landed, worker.peer_total_sent, worker.on_link), (Count(5, 5), Count(5, 5), Count(0, 0))
        )
        self.assertEqual(fed.on_link, Count(1, 1))

    def test_on_link_is_a_signed_difference_of_counters_that_wrap(self):
        # Channel 0: the peer's total has wrapped past 2^32 and is 2 ahead. Channel 1: the sender was read after its
        # peer, and one more completion landed in between, so it's 1 behind.
        result = self.l1_flows([0xFFFFFFFF, 7, 0, 0], [1, 6, 0, 0])
        self.assertEqual(result[WORKER].credits["completed"].on_link, Count(2, 2))
        self.assertEqual(result[FED].credits["completed"].on_link, Count(-1, -1))

    def test_a_peer_not_in_the_snapshot(self):
        credit = self.l1_flows([5, 6, 0, 0], None)[WORKER].credits["completed"]
        self.assertEqual(credit.total_landed, Count(5, 5))
        self.assertEqual(credit.peer_total_sent.not_observable, "the peer, M0/C1/W0, isn't in this snapshot")
        self.assertIn("needs both totals", credit.on_link.not_observable)

    def test_a_reference_that_disagrees_with_the_backing_is_refused(self):
        stream_references = manifest_with(
            lambda m: m["meshes"]["M0"]["credit_transport"]["vc0"].update(backing="l1_counter")
        )
        with self.assertRaisesRegex(
            ManifestError, r"credits\.completed is a stream register reference, but .* says l1_counter"
        ):
            flows([east_router(), sibling_router()], stream_references)

    def test_speedy_vc0_labels_its_batch(self):
        result = flows([east_router(), sibling_router()], manifest(speedy=True))
        self.assertEqual(result[WORKER].speedy_batch, 4)
        self.assertIsNone(flows([east_router(), sibling_router()])[WORKER].speedy_batch)


class FirstLevelAckTest(unittest.TestCase):
    def test_without_it_a_receiver_counts_waiting_packets(self):
        receiver = flows([east_router({f"{RECEIVER}/fields/pkts_sent": stream(3)}), sibling_router()])[RECEIVER]
        self.assertEqual(receiver.waiting, Count(3, 3))
        self.assertEqual(receiver.unacked.not_observable, "first-level ack is off on this VC")

    def test_with_it_a_receiver_counts_unacked_packets_and_the_worker_channel_has_acks(self):
        streams = {f"{RECEIVER}/fields/pkts_sent": stream(3), f"{WORKER}/credits/acked": stream(1)}
        result = flows([east_router(streams), sibling_router()], manifest(first_level_ack=True))
        self.assertEqual(result[RECEIVER].unacked, Count(3, 3))
        self.assertIn("first-level ack", result[RECEIVER].waiting.not_observable)
        acked = result[WORKER].credits["acked"]
        self.assertEqual((acked.frees, acked.pending), ("sender_slot", Count(1, 1)))


def manifest_with(change, base=None):
    changed = copy.deepcopy(base or manifest())
    change(changed)
    return changed


def east(manifest_):
    return manifest_["meshes"]["M0"]["chips"]["C0"]["routers"]["E0"]


class ManifestTest(unittest.TestCase):
    def test_values_the_rules_dont_know_are_refused(self):
        cases = {
            "an unknown backing": (
                lambda m: m["meshes"]["M0"]["credit_transport"]["vc0"].update(backing="mailbox"),
                "backing is 'mailbox'",
            ),
            "a renamed connection state": (
                lambda m: m["enums"]["ConnectionValue"]["values"].update(closing=3),
                r"ConnectionValue names .*'closing'",
            ),
            "an unknown credit type": (
                lambda m: east(m)["channels"]["senders"]["vc0"]["ch0"]["credits"].update(nacked=stream_ref(40)),
                r"credits names \['nacked'\]",
            ),
            "a producer with no edge to the channel": (
                lambda m: east(m)["channels"]["senders"]["vc0"]["ch1"].update(producer="M0/C1/W0"),
                "none of its intra_chip_downstream_edges lands on",
            ),
        }
        for name, (change, named) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ManifestError, named):
                flows([east_router(), sibling_router()], manifest_with(change))


if __name__ == "__main__":
    unittest.main()
