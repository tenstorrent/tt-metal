#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Standalone ack-channel consumer — stands in for the scheduler.

Attaches (as CONNECTOR) to the LayerAck InterProcessCounterChannel the prefill runner
creates in single-rank request mode, drains acks, and reports the total. Run it in a
third terminal alongside the runner (request mode) + prefill_h2d_producer.py.

The runner owns the channel at /tt_prefill_layer_acks_<service_id>; this connects to the
same name. Expected total = NUM_LAYERS * NUM_CHUNKS (default 61 * 11 = 671).

Usage:
    source python_env/bin/activate
    PREFILL_H2D_SERVICE_ID=ds_prefill \
    PREFILL_NUM_LAYERS=61 PREFILL_STANDALONE_NCHUNKS=11 \
      python3 -m models.demos.common.prefill.runners.prefill_layer_ack_consumer
"""

import os
import time

from loguru import logger

import ttnn

SERVICE_ID = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
NUM_CHUNKS = int(os.environ.get("PREFILL_STANDALONE_NCHUNKS", 11))
CONNECT_TIMEOUT_MS = int(os.environ.get("PREFILL_ACK_CONNECT_TIMEOUT_MS", 300_000))
IDLE_STOP_S = float(os.environ.get("PREFILL_ACK_IDLE_STOP_S", 30.0))


def main() -> None:
    shm_name = f"/tt_prefill_layer_acks_{SERVICE_ID}"
    expected = NUM_LAYERS * NUM_CHUNKS
    logger.info(f"[ack-consumer] connecting to {shm_name} (expecting {NUM_LAYERS} x {NUM_CHUNKS} = {expected} acks)")

    ch = ttnn.InterProcessCounterChannel.connect(shm_name, CONNECT_TIMEOUT_MS)
    logger.info(f"[ack-consumer] connected; clean_prior_shutdown={ch.had_clean_prior_shutdown()}")

    total = 0
    last_progress = time.monotonic()
    try:
        while True:
            n = ch.try_consume_all()
            if n:
                total += n
                last_progress = time.monotonic()
                logger.info(f"[ack-consumer] +{n} acks (total {total}/{expected})")
                if total >= expected:
                    logger.success(f"[ack-consumer] reached expected {expected} acks")
                    break
            else:
                if time.monotonic() - last_progress > IDLE_STOP_S:
                    logger.warning(
                        f"[ack-consumer] idle {IDLE_STOP_S}s with no new acks; stopping at {total}/{expected}"
                    )
                    break
                time.sleep(0.02)
    finally:
        logger.info(f"[ack-consumer] FINAL total acks = {total} (expected {expected})")
        ch.shutdown()

    if total != expected:
        raise SystemExit(f"ack count mismatch: got {total}, expected {expected}")


if __name__ == "__main__":
    main()
