# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test a long chain of tool invocations directly (no web server):
  text → audio → text → image → audio → text

Chain breakdown:
1. Text → TTS (SpeechT5) → Audio file
2. Audio → STT (Whisper) → Transcribed text
3. Text → Object Detection (OWL-ViT) with dummy image → Description
4. Description → TTS (SpeechT5) → Audio file
5. Audio → STT (Whisper) → Final text

Note: Skipping Stable Diffusion as it conflicts with other models in shared device mode.

Usage:
    cd /home/ubuntu/agentic/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/minimax_m2/agentic/tests/test_tool_chain_direct.py
"""

import os
import time

from loguru import logger


def log_step(step_num, total, description):
    logger.info(f"\n{'='*60}")
    logger.info(f"[{step_num}/{total}] {description}")
    logger.info("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("Multi-Modal Tool Chain Test (Direct)")
    logger.info("text → audio → text → (image) → audio → text")
    logger.info("=" * 60)

    # Import tools
    from models.demos.minimax_m2.agentic.loader import cleanup_models, load_all_models, open_n300_device

    # Open device
    logger.info("\n[INIT] Opening N300 mesh device...")
    mesh_device = open_n300_device(enable_fabric=True)
    logger.info(f"Device opened: {mesh_device.get_num_devices()} chips")

    try:
        # Load models (skip SD to avoid conflicts)
        logger.info("\n[INIT] Loading models (this takes ~3 minutes)...")
        start_load = time.time()
        models = load_all_models(
            mesh_device,
            load_llm=False,  # Skip LLM for faster loading
            load_whisper=True,
            load_speecht5=True,
            load_owlvit=True,
            load_bert=False,
            load_yunet=False,
            load_t5=False,
            load_sbert=False,
            load_sd=False,  # SD conflicts with other models
        )
        load_time = time.time() - start_load
        logger.info(f"Models loaded in {load_time:.1f}s")

        chain_results = []
        start_time = time.time()

        # =====================================================================
        # Step 1: Text → Audio (TTS)
        # =====================================================================
        log_step(1, 5, "TEXT → AUDIO (TTS with SpeechT5)")

        original_text = "Hello, I am testing a multi-modal chain on Tenstorrent hardware."
        logger.info(f"Input text: '{original_text}'")

        step_start = time.time()
        try:
            # Generate speech
            audio_path_1 = "/tmp/chain_audio_1.wav"
            models.speecht5.synthesize(original_text, audio_path_1)
            step_time = time.time() - step_start

            if os.path.exists(audio_path_1):
                file_size = os.path.getsize(audio_path_1)
                logger.info(f"✅ Generated audio: {audio_path_1} ({file_size} bytes)")
                logger.info(f"   Time: {step_time:.1f}s")
                chain_results.append(("text→audio", True, step_time))
            else:
                logger.error(f"❌ TTS failed - no audio generated")
                chain_results.append(("text→audio", False, step_time))
                audio_path_1 = None
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            import traceback

            traceback.print_exc()
            chain_results.append(("text→audio", False, 0))
            audio_path_1 = None

        # =====================================================================
        # Step 2: Audio → Text (STT)
        # =====================================================================
        log_step(2, 5, "AUDIO → TEXT (STT with Whisper)")

        if audio_path_1:
            logger.info(f"Input audio: {audio_path_1}")
            step_start = time.time()

            try:
                transcribed_1 = models.whisper.transcribe(audio_path_1)
                step_time = time.time() - step_start

                logger.info(f"✅ Transcribed: '{transcribed_1}'")
                logger.info(f"   Time: {step_time:.1f}s")
                chain_results.append(("audio→text", True, step_time))
            except Exception as e:
                logger.error(f"❌ STT error: {e}")
                import traceback

                traceback.print_exc()
                chain_results.append(("audio→text", False, 0))
                transcribed_1 = original_text
        else:
            logger.warning("⚠️ Skipping - no audio from previous step")
            transcribed_1 = original_text
            chain_results.append(("audio→text", None, 0))

        # =====================================================================
        # Step 3: Image → Text (Object Detection)
        # =====================================================================
        log_step(3, 5, "IMAGE → TEXT (Object Detection with OWL-ViT)")

        # Use a test image
        test_images = [
            "/home/ubuntu/agentic/tt-metal/models/demos/yolov4/images/coco_sample.jpg",
            "/home/ubuntu/agentic/tt-metal/models/tt_transformers/demo/sample_prompts/llama_models/dog.jpg",
        ]

        image_path = None
        for p in test_images:
            if os.path.exists(p):
                image_path = p
                break

        if image_path:
            logger.info(f"Input image: {image_path}")
            step_start = time.time()

            try:
                # Detect objects
                labels = ["person", "dog", "cat", "car", "bicycle", "tree", "building"]
                detections = models.owlvit.detect(image_path, labels)
                step_time = time.time() - step_start

                # Format description
                if detections:
                    desc_parts = []
                    for det in detections[:3]:  # Top 3
                        label = det.get("label", "object")
                        score = det.get("score", 0)
                        desc_parts.append(f"{label} with {score:.0%} confidence")
                    image_description = "I detected " + ", ".join(desc_parts) + " in the image."
                else:
                    image_description = "I did not detect any specific objects in the image."

                logger.info(f"✅ Detections: {len(detections)} objects")
                logger.info(f"   Description: '{image_description}'")
                logger.info(f"   Time: {step_time:.1f}s")
                chain_results.append(("image→text", True, step_time))
            except Exception as e:
                logger.error(f"❌ Detection error: {e}")
                import traceback

                traceback.print_exc()
                chain_results.append(("image→text", False, 0))
                image_description = "An image was processed."
        else:
            logger.warning("⚠️ No test image found - using placeholder")
            image_description = "A beautiful scene was described."
            chain_results.append(("image→text", None, 0))

        # =====================================================================
        # Step 4: Text → Audio (TTS again)
        # =====================================================================
        log_step(4, 5, "TEXT → AUDIO (TTS with SpeechT5)")

        logger.info(f"Input text: '{image_description}'")

        step_start = time.time()
        try:
            audio_path_2 = "/tmp/chain_audio_2.wav"
            models.speecht5.synthesize(image_description, audio_path_2)
            step_time = time.time() - step_start

            if os.path.exists(audio_path_2):
                file_size = os.path.getsize(audio_path_2)
                logger.info(f"✅ Generated audio: {audio_path_2} ({file_size} bytes)")
                logger.info(f"   Time: {step_time:.1f}s")
                chain_results.append(("text→audio(2)", True, step_time))
            else:
                logger.error(f"❌ TTS failed")
                chain_results.append(("text→audio(2)", False, step_time))
                audio_path_2 = None
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            import traceback

            traceback.print_exc()
            chain_results.append(("text→audio(2)", False, 0))
            audio_path_2 = None

        # =====================================================================
        # Step 5: Audio → Text (STT final)
        # =====================================================================
        log_step(5, 5, "AUDIO → TEXT (STT with Whisper)")

        if audio_path_2:
            logger.info(f"Input audio: {audio_path_2}")
            step_start = time.time()

            try:
                final_text = models.whisper.transcribe(audio_path_2)
                step_time = time.time() - step_start

                logger.info(f"✅ Final transcription: '{final_text}'")
                logger.info(f"   Time: {step_time:.1f}s")
                chain_results.append(("audio→text(2)", True, step_time))
            except Exception as e:
                logger.error(f"❌ STT error: {e}")
                chain_results.append(("audio→text(2)", False, 0))
                final_text = ""
        else:
            logger.warning("⚠️ Skipping - no audio from previous step")
            final_text = image_description
            chain_results.append(("audio→text(2)", None, 0))

        # =====================================================================
        # Summary
        # =====================================================================
        total_time = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("CHAIN RESULTS SUMMARY")
        logger.info("=" * 60)

        logger.info("\n Chain: text → audio → text → image → audio → text\n")

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
            logger.info(f"  {step_name:20} {status} ({duration:.1f}s)")

        logger.info(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
        logger.info(f"Total chain time: {total_time:.1f}s (excluding model load)")

        logger.info("\n" + "=" * 60)
        if failed == 0 and skipped == 0:
            logger.info("SUCCESS: Full chain completed!")
        elif failed == 0:
            logger.info("PARTIAL SUCCESS: Chain completed with skips")
        else:
            logger.info("ISSUES: Some steps failed")
        logger.info("=" * 60)

        return failed == 0

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        logger.info("\nCleaning up...")
        cleanup_models(models)
        import ttnn

        ttnn.close_mesh_device(mesh_device)
        logger.info("Done!")


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
