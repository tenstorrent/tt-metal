# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test a long chain of tool invocations:
  text → audio → text → image → audio → text

Chain breakdown:
1. Text → TTS (SpeechT5) → Audio file
2. Audio → STT (Whisper) → Transcribed text
3. Text → Image Gen (Stable Diffusion) → Image file
4. Image → Object Detection (OWL-ViT) → Description text
5. Description → TTS (SpeechT5) → Audio file
6. Audio → STT (Whisper) → Final text

Usage:
    cd /home/ubuntu/agentic/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/agentic/tests/test_tool_chain.py
"""

import os
import time

import requests

BASE_URL = "http://localhost:7010"
TIMEOUT = 300  # 5 minutes for image generation


def log_step(step_num, total, description):
    print(f"\n{'='*60}")
    print(f"[{step_num}/{total}] {description}")
    print("=" * 60)


def wait_for_server():
    """Wait for server to be ready."""
    print("Waiting for server to be ready...")
    for i in range(120):  # 10 minutes max
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("models_loaded"):
                print(f"Server ready after {i*5}s")
                return True
        except Exception:
            pass
        time.sleep(5)
    print("Server not ready after 10 minutes")
    return False


def query_llm(text: str, image_path: str = None, audio_path: str = None, want_audio: bool = False) -> dict:
    """Send a query to the server and get response."""
    payload = {
        "text": text,
        "want_audio_response": want_audio,
    }
    if image_path:
        payload["image_path"] = image_path
    if audio_path:
        payload["audio_path"] = audio_path

    resp = requests.post(f"{BASE_URL}/query", json=payload, timeout=TIMEOUT)
    return resp.json()


def direct_tts(text: str) -> str:
    """Directly call TTS endpoint."""
    resp = requests.post(f"{BASE_URL}/tts", json={"text": text}, timeout=120)
    result = resp.json()
    return result.get("audio_path")


def direct_stt(audio_path: str) -> str:
    """Directly call STT endpoint."""
    resp = requests.post(f"{BASE_URL}/stt", json={"audio_path": audio_path}, timeout=120)
    result = resp.json()
    return result.get("transcription", result.get("text", ""))


def direct_image_gen(prompt: str) -> str:
    """Directly call image generation endpoint."""
    resp = requests.post(f"{BASE_URL}/generate-image", json={"prompt": prompt}, timeout=TIMEOUT)
    result = resp.json()
    return result.get("image_path")


def direct_detect(image_path: str, labels: list) -> dict:
    """Directly call object detection endpoint."""
    resp = requests.post(f"{BASE_URL}/detect", json={"image_path": image_path, "labels": labels}, timeout=120)
    return resp.json()


def main():
    print("=" * 60)
    print("Multi-Modal Tool Chain Test")
    print("text → audio → text → image → audio → text")
    print("=" * 60)

    if not wait_for_server():
        print("FAILED: Server not ready")
        return False

    chain_results = []
    start_time = time.time()

    # =========================================================================
    # Step 1: Text → Audio (TTS)
    # =========================================================================
    log_step(1, 6, "TEXT → AUDIO (TTS with SpeechT5)")

    original_text = "Hello, I am testing a multi-modal chain on Tenstorrent hardware."
    print(f"Input text: '{original_text}'")

    step_start = time.time()
    try:
        # Use query endpoint with want_audio_response
        result = query_llm(f"Please repeat exactly: {original_text}", want_audio=True)
        audio_path_1 = result.get("audio_path")

        if not audio_path_1:
            # Try direct TTS
            print("Trying direct TTS endpoint...")
            audio_path_1 = direct_tts(original_text)

        step_time = time.time() - step_start

        if audio_path_1 and os.path.exists(audio_path_1):
            print(f"✅ Generated audio: {audio_path_1}")
            print(f"   Time: {step_time:.1f}s")
            chain_results.append(("text→audio", True, step_time))
        else:
            print(f"❌ TTS failed - no audio generated")
            chain_results.append(("text→audio", False, step_time))
            audio_path_1 = None
    except Exception as e:
        print(f"❌ TTS error: {e}")
        chain_results.append(("text→audio", False, 0))
        audio_path_1 = None

    # =========================================================================
    # Step 2: Audio → Text (STT)
    # =========================================================================
    log_step(2, 6, "AUDIO → TEXT (STT with Whisper)")

    if audio_path_1:
        print(f"Input audio: {audio_path_1}")
        step_start = time.time()

        try:
            result = query_llm("What did I say in this audio? Just transcribe it.", audio_path=audio_path_1)
            transcribed_1 = result.get("text", "")
            step_time = time.time() - step_start

            print(f"✅ Transcribed: '{transcribed_1[:100]}...'")
            print(f"   Time: {step_time:.1f}s")
            chain_results.append(("audio→text", True, step_time))
        except Exception as e:
            print(f"❌ STT error: {e}")
            chain_results.append(("audio→text", False, 0))
            transcribed_1 = original_text  # Fallback
    else:
        print("⚠️ Skipping - no audio from previous step")
        transcribed_1 = original_text
        chain_results.append(("audio→text", None, 0))

    # =========================================================================
    # Step 3: Text → Image (Stable Diffusion)
    # =========================================================================
    log_step(3, 6, "TEXT → IMAGE (Stable Diffusion)")

    image_prompt = "A beautiful sunset over mountains with purple sky"
    print(f"Image prompt: '{image_prompt}'")

    step_start = time.time()
    try:
        result = query_llm(f"Generate an image of: {image_prompt}")
        image_path = result.get("image_path")

        if not image_path:
            # Try direct endpoint
            print("Trying direct image generation endpoint...")
            image_path = direct_image_gen(image_prompt)

        step_time = time.time() - step_start

        if image_path and os.path.exists(image_path):
            print(f"✅ Generated image: {image_path}")
            print(f"   Time: {step_time:.1f}s")
            chain_results.append(("text→image", True, step_time))
        else:
            print(f"❌ Image generation failed")
            chain_results.append(("text→image", False, step_time))
            image_path = None
    except Exception as e:
        print(f"❌ Image gen error: {e}")
        chain_results.append(("text→image", False, 0))
        image_path = None

    # =========================================================================
    # Step 4: Image → Text (Object Detection / Description)
    # =========================================================================
    log_step(4, 6, "IMAGE → TEXT (Object Detection with OWL-ViT)")

    if image_path:
        print(f"Input image: {image_path}")
        step_start = time.time()

        try:
            result = query_llm("What objects do you see in this image? Describe them briefly.", image_path=image_path)
            image_description = result.get("text", "")
            step_time = time.time() - step_start

            print(f"✅ Description: '{image_description[:100]}...'")
            print(f"   Time: {step_time:.1f}s")
            chain_results.append(("image→text", True, step_time))
        except Exception as e:
            print(f"❌ Detection error: {e}")
            chain_results.append(("image→text", False, 0))
            image_description = "A beautiful image"
    else:
        print("⚠️ Skipping - no image from previous step")
        image_description = "A beautiful sunset over mountains"
        chain_results.append(("image→text", None, 0))

    # =========================================================================
    # Step 5: Text → Audio (TTS again)
    # =========================================================================
    log_step(5, 6, "TEXT → AUDIO (TTS with SpeechT5)")

    # Use a shortened version for TTS
    tts_text = image_description[:200] if len(image_description) > 200 else image_description
    print(f"Input text: '{tts_text[:80]}...'")

    step_start = time.time()
    try:
        result = query_llm(f"Please say: {tts_text}", want_audio=True)
        audio_path_2 = result.get("audio_path")

        if not audio_path_2:
            print("Trying direct TTS endpoint...")
            audio_path_2 = direct_tts(tts_text)

        step_time = time.time() - step_start

        if audio_path_2 and os.path.exists(audio_path_2):
            print(f"✅ Generated audio: {audio_path_2}")
            print(f"   Time: {step_time:.1f}s")
            chain_results.append(("text→audio(2)", True, step_time))
        else:
            print(f"❌ TTS failed")
            chain_results.append(("text→audio(2)", False, step_time))
            audio_path_2 = None
    except Exception as e:
        print(f"❌ TTS error: {e}")
        chain_results.append(("text→audio(2)", False, 0))
        audio_path_2 = None

    # =========================================================================
    # Step 6: Audio → Text (STT final)
    # =========================================================================
    log_step(6, 6, "AUDIO → TEXT (STT with Whisper)")

    if audio_path_2:
        print(f"Input audio: {audio_path_2}")
        step_start = time.time()

        try:
            result = query_llm("Transcribe this audio.", audio_path=audio_path_2)
            final_text = result.get("text", "")
            step_time = time.time() - step_start

            print(f"✅ Final transcription: '{final_text[:100]}...'")
            print(f"   Time: {step_time:.1f}s")
            chain_results.append(("audio→text(2)", True, step_time))
        except Exception as e:
            print(f"❌ STT error: {e}")
            chain_results.append(("audio→text(2)", False, 0))
            final_text = ""
    else:
        print("⚠️ Skipping - no audio from previous step")
        final_text = tts_text
        chain_results.append(("audio→text(2)", None, 0))

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("CHAIN RESULTS SUMMARY")
    print("=" * 60)

    print("\n Chain: text → audio → text → image → audio → text\n")

    passed = 0
    failed = 0
    skipped = 0

    for step_name, success, duration in chain_results:
        if success is True:
            status = "✅ PASS"
            passed += 1
        elif success is False:
            status = "❌ FAIL"
            failed += 1
        else:
            status = "⚠️ SKIP"
            skipped += 1
        print(f"  {step_name:20} {status} ({duration:.1f}s)")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total time: {total_time:.1f}s")

    print("\n" + "=" * 60)
    if failed == 0 and skipped == 0:
        print("SUCCESS: Full chain completed!")
    elif failed == 0:
        print("PARTIAL SUCCESS: Chain completed with skips")
    else:
        print("ISSUES: Some steps failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
