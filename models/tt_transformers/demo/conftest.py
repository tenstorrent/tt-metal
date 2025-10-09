# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_transformers.tt.model_config import parse_optimizations


# These inputs override the default inputs used by simple_text_demo.py. Check the main demo to see the default values.
def pytest_addoption(parser):
    parser.addoption("--input_prompts", action="store", help="input prompts json file")
    parser.addoption("--instruct", action="store", type=int, help="Use instruct weights")
    parser.addoption("--repeat_batches", action="store", type=int, help="Number of consecutive batches of users to run")
    parser.addoption("--max_seq_len", action="store", type=int, help="Maximum context length supported by the model")
    parser.addoption("--batch_size", action="store", type=int, help="Number of users in a batch ")
    parser.addoption(
        "--max_generated_tokens", action="store", type=int, help="Maximum number of tokens to generate for each user"
    )
    parser.addoption("--data_parallel", action="store", type=int, help="Number of data parallel workers")
    parser.addoption(
        "--paged_attention", action="store", type=bool, help="Whether to use paged attention or default attention"
    )
    parser.addoption("--page_params", action="store", type=dict, help="Page parameters for paged attention")
    parser.addoption("--sampling_params", action="store", type=dict, help="Sampling parameters for decoding")
    parser.addoption(
        "--stop_at_eos", action="store", type=int, help="Whether to stop decoding when the model generates an EoS token"
    )
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
    parser.addoption(
        "--decoder_config_file",
        action="store",
        default=None,
        type=str,
        help="Provide a JSON file defining per-decoder precision and fidelity settings",
    )
    parser.addoption(
        "--token_accuracy",
        action="store",
        default=False,
        type=bool,
        help="Whether to compute top1 and top5 exact token matching accuracy",
    )
    parser.addoption(
        "--stress_test",
        action="store",
        default=False,
        type=bool,
        help="Run stress test (same decode iteration over a large number of iterations",
    )
    parser.addoption(
        "--enable_trace",
        action="store",
        default=None,
        type=bool,
        help="Whether to enable tracing",
    )


@pytest.fixture(scope="function", autouse=True)
def clear_program_cache_after_test(request, mesh_device):
    """Automatically clear program cache after each test to prevent memory accumulation."""
    import gc
    import socket
    import struct
    import sys
    import time

    import ttnn

    yield  # Run the test

    # Cleanup after test
    if isinstance(mesh_device, ttnn.MeshDevice):
        # Use print() to bypass pytest capture and ensure visibility
        print("\n" + "=" * 80, file=sys.stderr)
        print("CLEANUP FIXTURE: Starting post-test cleanup...", file=sys.stderr)

        # Force garbage collection first
        print("CLEANUP FIXTURE: Running garbage collection...", file=sys.stderr)
        gc.collect()

        # Clear program cache
        print("CLEANUP FIXTURE: Clearing program cache...", file=sys.stderr)
        try:
            mesh_device.disable_and_clear_program_cache()
            print("✓ Program cache cleared", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to clear program cache: {e}", file=sys.stderr)

        # Force another GC after cache clear
        gc.collect()

        # Close the mesh device to free all buffers (including TRACE buffers)
        print("CLEANUP FIXTURE: Closing mesh_device to free all buffers...", file=sys.stderr)
        try:
            ttnn.close_mesh_device(mesh_device)
            print("✓ mesh_device closed - all buffers (including TRACE) should be freed", file=sys.stderr)
        except Exception as e:
            print(f"❌ Failed to close mesh_device: {e}", file=sys.stderr)

        # Force GC after close
        gc.collect()

        # Wait a moment for deallocations to propagate
        print("CLEANUP FIXTURE: Waiting for deallocations to propagate...", file=sys.stderr)
        time.sleep(2)  # Increased from 1 to 2 seconds

        # Request dump of remaining buffers from allocation server
        try:
            print("CLEANUP FIXTURE: Requesting buffer dump from allocation server...", file=sys.stderr)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)  # 5 second timeout
            sock.connect("/tmp/tt_allocation_server.sock")

            # Send DUMP_REMAINING message (type=5)
            msg = struct.pack("B3xiQB3xiQQ4Q", 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            sock.send(msg)
            sock.close()

            print("✓ Buffer dump requested - CHECK ALLOCATION SERVER OUTPUT", file=sys.stderr)
            print("  (The server will print remaining buffers to its terminal)", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Could not request buffer dump: {e}", file=sys.stderr)
            print("  (Allocation server may not be running)", file=sys.stderr)

        print("✓ Post-test cleanup complete", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)
