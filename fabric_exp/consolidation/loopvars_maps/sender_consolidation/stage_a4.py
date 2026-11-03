"""stage_a4.py: v4txa4 = v4txa3 + Parts A4-A7 of ROUTER_HELPERS_REVIEW.md (user-approved 2026-11-03):
  - A4: RoundCounters keeps what every loop's round needs (heartbeat, idle rounds); the sender's fields (send_yielded,
    sent_at_round_start) go to sender::SenderRound, which holds the counters first (the same layout for the sender);
  - A5: local_inline_write's byte enables are INLINE_WRITE_ALL_BYTES; the scatter chunk mask is named as the library
    names its own (CHUNK_ENCODING_MASK is local to the library's delivery function, so it is not shared);
    count_local_writes says it is the one place the router touches the API's counters. The trimming capture's
    send-type record stays one call per arm: hoisted before the dispatch it changed the receive loop on capture images
    (the review asked for it only if identity held);
  - A6: telemetry's verb pair note_sent / note_received; words_of(bytes); perf_buffer();
  - A7: one ChipCoordinates factory, this_chip(), used by receiver::run and single::run."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_lib import make
make("v4txa3", "v4txa4", "a4.patch")
