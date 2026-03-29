#!/usr/bin/env python3
"""
Video Verification Test Runner for Molmo2 vLLM Server

Runs all video tests from test.jsonl against the vLLM server and logs results.
Compilation/warmup happens once at server startup, then all tests run back-to-back.

Usage:
    python run_video_tests.py [--output results.jsonl] [--server-url http://localhost:8000]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests


def run_single_test(server_url: str, test_case: dict, test_idx: int) -> dict:
    """Run a single video test and return the result."""
    start_time = time.time()

    # Extract video URL and prompt for logging
    content = test_case["messages"][0]["content"]
    video_url = None
    prompt_text = None
    for item in content:
        if item.get("type") == "video_url":
            video_url = item["video_url"]["url"]
        elif item.get("type") == "text":
            prompt_text = item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]

    # Build request payload
    # Always use the server's model name regardless of test case
    payload = {
        "model": "allenai/Molmo2-8B",  # Server model name
        "messages": test_case["messages"],
        "max_tokens": test_case.get("max_tokens", 16),
        "temperature": test_case.get("temperature", 0),
    }

    result = {
        "test_idx": test_idx,
        "video_url": video_url,
        "prompt": prompt_text,
        "status": None,
        "response": None,
        "error": None,
        "latency_ms": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minute timeout per request
        )

        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)

        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            result["status"] = "success"
            result["response"] = content
            result["usage"] = resp_json.get("usage", {})
        else:
            result["status"] = "error"
            result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        result["status"] = "timeout"
        result["error"] = "Request timed out after 300 seconds"
        result["latency_ms"] = 300000
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["latency_ms"] = round((time.time() - start_time) * 1000, 2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run video verification tests against Molmo2 vLLM server")
    parser.add_argument("--input", type=str, default="test.jsonl", help="Input JSONL file with test cases")
    parser.add_argument("--output", type=str, default="video_test_results.jsonl", help="Output JSONL file for results")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--start", type=int, default=0, help="Start from test index (0-based)")
    parser.add_argument("--count", type=int, default=None, help="Number of tests to run (default: all)")
    args = parser.parse_args()

    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)

    # Load test cases
    print(f"Loading tests from: {input_path}")
    test_cases = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    total_tests = len(test_cases)
    print(f"Loaded {total_tests} test cases")

    # Apply start/count filters
    end_idx = args.start + args.count if args.count else total_tests
    test_cases = test_cases[args.start : end_idx]
    print(f"Running tests {args.start} to {end_idx - 1} ({len(test_cases)} tests)")

    # Check server health
    print(f"\nChecking server at {args.server_url}...")
    try:
        health_resp = requests.get(f"{args.server_url}/health", timeout=10)
        if health_resp.status_code != 200:
            print(f"WARNING: Server health check returned {health_resp.status_code}")
        else:
            print("Server is healthy")
    except Exception as e:
        print(f"ERROR: Cannot connect to server: {e}")
        print("Make sure the vLLM server is running.")
        return

    # Run tests
    print(f"\n{'='*60}")
    print(f"Starting video verification tests")
    print(f"Output will be written to: {output_path}")
    print(f"{'='*60}\n")

    results = []
    success_count = 0
    error_count = 0
    timeout_count = 0
    total_latency = 0

    start_time = time.time()

    with open(output_path, "w") as out_f:
        for i, test_case in enumerate(test_cases):
            test_idx = args.start + i
            print(f"[{test_idx + 1}/{total_tests}] Running test...", end=" ", flush=True)

            result = run_single_test(args.server_url, test_case, test_idx)
            results.append(result)

            # Write result immediately (streaming output)
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            # Update counters
            if result["status"] == "success":
                success_count += 1
                total_latency += result["latency_ms"]
                print(f"SUCCESS ({result['latency_ms']:.0f}ms) - {result['response'][:50]}...")
            elif result["status"] == "timeout":
                timeout_count += 1
                print(f"TIMEOUT")
            else:
                error_count += 1
                print(f"ERROR: {result['error'][:50]}...")

    # Summary
    elapsed = time.time() - start_time
    avg_latency = total_latency / success_count if success_count > 0 else 0

    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total tests run: {len(test_cases)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Timeouts: {timeout_count}")
    print(f"Average latency (success): {avg_latency:.0f}ms")
    print(f"Total elapsed time: {elapsed:.1f}s")
    print(f"\nResults saved to: {output_path}")

    # Also save summary
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(input_path),
        "total_tests": len(test_cases),
        "success": success_count,
        "errors": error_count,
        "timeouts": timeout_count,
        "avg_latency_ms": round(avg_latency, 2),
        "total_elapsed_s": round(elapsed, 2),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
