#!/usr/bin/env python3
"""
Comprehensive Molmo2 Test Suite

Tests all modalities (text, image, video) across:
- Batch 1 (demo.py)
- Batch 32 (demo.py)
- Docker server batch 1 (sequential requests)
- Docker server batch 32 (concurrent requests)

Usage:
    # Run all tests (requires server for docker tests)
    python models/demos/molmo2/run_comprehensive_tests.py --all

    # Run only demo tests (no server needed)
    python models/demos/molmo2/run_comprehensive_tests.py --demo-only

    # Run only server tests
    python models/demos/molmo2/run_comprehensive_tests.py --server-only
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).parent
DEMO_DIR = SCRIPT_DIR / "demo"
RESULTS_DIR = SCRIPT_DIR / "verification"

# Test prompts for different modalities
TEXT_PROMPTS = [
    "What is 2 + 2?",
    "Name three primary colors.",
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
]

TEXT_PROMPTS_32 = [f"User {i}: What is {i} plus {i+1}?" for i in range(32)]

IMAGE_PROMPTS = [
    {"image": str(DEMO_DIR / "dog.jpg"), "prompt": "<|image|> Describe this image in detail."},
    {"image": str(DEMO_DIR / "dog.jpg"), "prompt": "<|image|> What breed of dog is this?"},
    {"image": str(DEMO_DIR / "dog.jpg"), "prompt": "<|image|> What is the dog doing?"},
    {"image": str(DEMO_DIR / "dog.jpg"), "prompt": "<|image|> What colors do you see?"},
]

# Use a short video for testing
VIDEO_URL = "https://storage.googleapis.com/oe-training-public/molmo2-eval-media/85682eb97ff9a6111ac0be7c1fdd087a37d496d53f5771b5922a22265a8ee25f.mp4"
VIDEO_PROMPTS = [
    "<|video|> What is happening in this video?",
    "<|video|> Describe the main action in this video.",
]


def run_demo_test(batch_size: int, modality: str, use_trace: bool = True) -> dict:
    """Run demo.py test for a specific batch size and modality."""
    result = {
        "test": f"demo_batch{batch_size}_{modality}",
        "batch_size": batch_size,
        "modality": modality,
        "use_trace": use_trace,
        "status": None,
        "output": None,
        "coherent": None,
        "error": None,
        "duration_s": None,
    }

    start_time = time.time()

    try:
        cmd = [
            "python",
            "-m",
            "models.demos.molmo2.demo.demo",
            "--batch-size",
            str(batch_size),
        ]

        if use_trace:
            cmd.append("--use-decode-trace")

        if modality == "text":
            # Create temp prompts file for text
            prompts = TEXT_PROMPTS if batch_size == 1 else TEXT_PROMPTS_32[:batch_size]
            prompts_file = RESULTS_DIR / f"temp_text_prompts_b{batch_size}.json"
            with open(prompts_file, "w") as f:
                json.dump([{"prompt": p} for p in prompts], f)
            cmd.extend(["--input-file", str(prompts_file)])

        elif modality == "image":
            # Use existing image prompts file or create one
            prompts = IMAGE_PROMPTS[: min(batch_size, len(IMAGE_PROMPTS))]
            prompts_file = RESULTS_DIR / f"temp_image_prompts_b{batch_size}.json"
            with open(prompts_file, "w") as f:
                json.dump(prompts, f)
            cmd.extend(["--input-file", str(prompts_file)])

        elif modality == "video":
            # Video test - use single video with decode trace
            cmd.extend(["--video", VIDEO_URL, "--prompt", VIDEO_PROMPTS[0]])
            cmd.append("--max-tokens")
            cmd.append("50")

        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(SCRIPT_DIR.parent.parent.parent),  # tt-metal root
        )

        result["output"] = proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout
        result["error"] = proc.stderr[-1000:] if proc.stderr else None
        result["status"] = "success" if proc.returncode == 0 else "failed"

        # Check output coherence
        if proc.returncode == 0:
            result["coherent"] = check_output_coherence(proc.stdout, modality, batch_size)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Test timed out after 600 seconds"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    result["duration_s"] = round(time.time() - start_time, 2)
    return result


def check_output_coherence(output: str, modality: str, batch_size: int) -> dict:
    """Check if the output is coherent based on modality."""
    coherence = {
        "is_coherent": True,
        "issues": [],
        "samples": [],
    }

    # Extract generated text from output
    lines = output.split("\n")
    generated_lines = [l for l in lines if "Generated:" in l or "Response:" in l or "Output:" in l]

    if not generated_lines:
        # Try to find output another way
        for i, line in enumerate(lines):
            if "tok/s" in line.lower() or "tokens" in line.lower():
                # Look at surrounding lines for output
                context = lines[max(0, i - 5) : min(len(lines), i + 5)]
                coherence["samples"] = context[:3]
                break
    else:
        coherence["samples"] = generated_lines[:3]

    # Check for common incoherence patterns
    output_lower = output.lower()

    # Repetition check
    if output_lower.count("...") > 10 or "around around" in output_lower:
        coherence["is_coherent"] = False
        coherence["issues"].append("Excessive repetition detected")

    # Garbage output check
    if len(set(output_lower.split())) < 20 and len(output) > 500:
        coherence["is_coherent"] = False
        coherence["issues"].append("Low vocabulary diversity (possible garbage)")

    # Check for correct math responses in text batch test
    if modality == "text" and batch_size > 1:
        # Look for correct answers in output
        correct_count = 0
        for i in range(min(batch_size, 32)):
            expected = i + (i + 1)
            if str(expected) in output:
                correct_count += 1

        accuracy = correct_count / batch_size if batch_size > 0 else 0
        coherence["accuracy"] = f"{correct_count}/{batch_size} ({accuracy*100:.0f}%)"
        if accuracy < 0.5:
            coherence["is_coherent"] = False
            coherence["issues"].append(f"Low accuracy: only {correct_count}/{batch_size} correct")

    return coherence


def run_server_test(batch_size: int, modality: str, server_url: str = "http://localhost:8000") -> dict:
    """Run server API test for a specific batch size and modality."""
    result = {
        "test": f"server_batch{batch_size}_{modality}",
        "batch_size": batch_size,
        "modality": modality,
        "server_url": server_url,
        "status": None,
        "responses": [],
        "coherent": None,
        "error": None,
        "duration_s": None,
        "throughput": None,
    }

    start_time = time.time()

    try:
        # Check server health first
        health = requests.get(f"{server_url}/health", timeout=10)
        if health.status_code != 200:
            result["status"] = "server_unhealthy"
            result["error"] = f"Server health check failed: {health.status_code}"
            return result

        # Build requests based on modality
        if modality == "text":
            prompts = TEXT_PROMPTS if batch_size == 1 else TEXT_PROMPTS_32[:batch_size]
            payloads = [
                {
                    "model": "allenai/Molmo2-8B",
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 50,
                    "temperature": 0,
                }
                for p in prompts
            ]

        elif modality == "image":
            prompts = IMAGE_PROMPTS[: min(batch_size, len(IMAGE_PROMPTS))]
            # For server, we need to use base64 or URL for images
            # Using text description as fallback
            payloads = [
                {
                    "model": "allenai/Molmo2-8B",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": p["prompt"]},
                                {"type": "image_url", "image_url": {"url": f"file://{p['image']}"}},
                            ],
                        }
                    ],
                    "max_tokens": 100,
                    "temperature": 0,
                }
                for p in prompts
            ]

        elif modality == "video":
            payloads = [
                {
                    "model": "allenai/Molmo2-8B",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": VIDEO_PROMPTS[i % len(VIDEO_PROMPTS)]},
                                {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                            ],
                        }
                    ],
                    "max_tokens": 50,
                    "temperature": 0,
                }
                for i in range(min(batch_size, 4))  # Limit video tests
            ]

        print(f"\n{'='*60}")
        print(f"Running server test: batch={batch_size}, modality={modality}")
        print(f"Sending {len(payloads)} requests...")
        print(f"{'='*60}")

        # Send requests (concurrent for batch > 1)
        import concurrent.futures

        def send_request(payload):
            resp_start = time.time()
            try:
                resp = requests.post(
                    f"{server_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=300,
                )
                resp_time = time.time() - resp_start
                if resp.status_code == 200:
                    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {"status": "success", "response": content, "latency_ms": resp_time * 1000}
                else:
                    return {"status": "error", "error": f"HTTP {resp.status_code}", "latency_ms": resp_time * 1000}
            except Exception as e:
                return {"status": "error", "error": str(e), "latency_ms": (time.time() - resp_start) * 1000}

        if batch_size > 1:
            # Concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                responses = list(executor.map(send_request, payloads))
        else:
            # Sequential for batch 1
            responses = [send_request(p) for p in payloads]

        result["responses"] = responses

        # Calculate stats
        successes = [r for r in responses if r["status"] == "success"]
        result["success_count"] = len(successes)
        result["total_count"] = len(responses)

        if successes:
            avg_latency = sum(r["latency_ms"] for r in successes) / len(successes)
            result["avg_latency_ms"] = round(avg_latency, 2)
            result["status"] = "success"

            # Check coherence
            result["coherent"] = check_server_output_coherence(responses, modality, batch_size)
        else:
            result["status"] = "failed"
            result["error"] = "All requests failed"

    except requests.exceptions.ConnectionError:
        result["status"] = "connection_error"
        result["error"] = "Cannot connect to server"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    result["duration_s"] = round(time.time() - start_time, 2)
    return result


def check_server_output_coherence(responses: list, modality: str, batch_size: int) -> dict:
    """Check coherence of server responses."""
    coherence = {
        "is_coherent": True,
        "issues": [],
        "samples": [],
    }

    successes = [r for r in responses if r["status"] == "success"]
    if not successes:
        coherence["is_coherent"] = False
        coherence["issues"].append("No successful responses")
        return coherence

    coherence["samples"] = [r["response"][:100] for r in successes[:3]]

    # Check for text correctness
    if modality == "text" and batch_size > 1:
        correct_count = 0
        for i, resp in enumerate(responses):
            if resp["status"] == "success":
                expected = i + (i + 1)
                if str(expected) in resp["response"]:
                    correct_count += 1

        accuracy = correct_count / len(responses) if responses else 0
        coherence["accuracy"] = f"{correct_count}/{len(responses)} ({accuracy*100:.0f}%)"
        if accuracy < 0.5:
            coherence["is_coherent"] = False
            coherence["issues"].append(f"Low accuracy: {correct_count}/{len(responses)} correct")

    # Check for repetitive/garbage output
    for resp in successes:
        text = resp["response"].lower()
        if text.count("...") > 5 or "around around" in text:
            coherence["is_coherent"] = False
            coherence["issues"].append("Repetitive output detected")
            break

    return coherence


def print_result(result: dict):
    """Print a test result in a readable format."""
    status_emoji = "✓" if result["status"] == "success" else "✗"
    coherent_status = ""
    if result.get("coherent"):
        coherent_emoji = "✓" if result["coherent"]["is_coherent"] else "⚠"
        coherent_status = f" | Coherent: {coherent_emoji}"
        if result["coherent"].get("accuracy"):
            coherent_status += f" ({result['coherent']['accuracy']})"

    print(f"{status_emoji} {result['test']}: {result['status']} ({result.get('duration_s', '?')}s){coherent_status}")

    if result.get("error"):
        print(f"   Error: {result['error'][:100]}")

    if result.get("coherent") and result["coherent"].get("issues"):
        for issue in result["coherent"]["issues"]:
            print(f"   ⚠ {issue}")

    if result.get("coherent") and result["coherent"].get("samples"):
        print(f"   Sample outputs:")
        for sample in result["coherent"]["samples"][:2]:
            print(f"     - {sample[:80]}...")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Molmo2 test suite")
    parser.add_argument("--all", action="store_true", help="Run all tests (demo + server)")
    parser.add_argument("--demo-only", action="store_true", help="Run only demo tests (no server needed)")
    parser.add_argument("--server-only", action="store_true", help="Run only server tests")
    parser.add_argument("--server-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    args = parser.parse_args()

    if not args.all and not args.demo_only and not args.server_only:
        args.demo_only = True  # Default to demo tests

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {},
    }

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("MOLMO2 COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Started: {results['timestamp']}")

    # Demo tests
    if args.all or args.demo_only:
        print("\n" + "-" * 70)
        print("DEMO TESTS (demo.py)")
        print("-" * 70)

        # Batch 1 tests
        for modality in ["text", "image", "video"]:
            result = run_demo_test(batch_size=1, modality=modality, use_trace=True)
            results["tests"].append(result)
            print_result(result)

        # Batch 32 tests (text and image only - video is single-input)
        for modality in ["text", "image"]:
            batch = 32 if modality == "text" else 4  # Limit image batch for memory
            result = run_demo_test(batch_size=batch, modality=modality, use_trace=True)
            results["tests"].append(result)
            print_result(result)

    # Server tests
    if args.all or args.server_only:
        print("\n" + "-" * 70)
        print("SERVER TESTS (vLLM API)")
        print("-" * 70)

        # Check if server is running
        try:
            health = requests.get(f"{args.server_url}/health", timeout=5)
            server_available = health.status_code == 200
        except:
            server_available = False

        if not server_available:
            print(f"⚠ Server not available at {args.server_url}")
            print("  Start server with: cd tt-inference-server && python run.py --model Molmo2-8B --tt-device t3k")
            results["summary"]["server_tests"] = "skipped - server unavailable"
        else:
            # Batch 1 tests
            for modality in ["text", "image", "video"]:
                result = run_server_test(batch_size=1, modality=modality, server_url=args.server_url)
                results["tests"].append(result)
                print_result(result)

            # Batch 32 tests
            for modality in ["text", "image", "video"]:
                batch = 32 if modality == "text" else 4  # Limit batch for image/video
                result = run_server_test(batch_size=batch, modality=modality, server_url=args.server_url)
                results["tests"].append(result)
                print_result(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(results["tests"])
    success = len([t for t in results["tests"] if t["status"] == "success"])
    coherent = len([t for t in results["tests"] if t.get("coherent", {}).get("is_coherent", False)])

    results["summary"]["total_tests"] = total
    results["summary"]["successful"] = success
    results["summary"]["coherent_outputs"] = coherent

    print(f"Total tests: {total}")
    print(f"Successful: {success}/{total} ({success/total*100:.0f}%)")
    print(f"Coherent outputs: {coherent}/{total} ({coherent/total*100:.0f}%)")

    # Save results
    output_file = args.output or str(
        RESULTS_DIR / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Return exit code
    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
