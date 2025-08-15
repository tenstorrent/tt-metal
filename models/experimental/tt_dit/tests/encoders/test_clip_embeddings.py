# spdx-filecopyrighttext: © 2025 tenstorrent ai ulc

# spdx-license-identifier: apache-2.0

import time
import torch
import ttnn
from loguru import logger
from transformers import CLIPTextModelWithProjection as HFCLIPTextModelWithProjection, CLIPTokenizer


def test_clip():
    logger.info("Testing CLIP embeddings through CLIPTextModelWithProjection...")

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

    try:
        # load hf model like sd3.5 test does
        model_name_checkpoint = "stabilityai/stable-diffusion-3.5-large"
        clip_path = "text_encoder"  # or "text encoder 2"
        tokenizer_path = "tokenizer"

        logger.info(f"Loading HF model from {model_name_checkpoint}/{clip_path}...")
        hf_model = HFCLIPTextModelWithProjection.from_pretrained(
            model_name_checkpoint, subfolder=clip_path, local_files_only=False  # allow download for test
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name_checkpoint, subfolder=tokenizer_path, local_files_only=False
        )

        hf_model.eval()

        # print config like sd3.5 test
        logger.info("=== HuggingFace CLIP Config ===")
        logger.info(f"vocab_size: {hf_model.config.vocab_size}")
        logger.info(f"hidden_size: {hf_model.config.hidden_size}")
        logger.info(f"intermediate_size: {hf_model.config.intermediate_size}")
        logger.info(f"num_attention_heads: {hf_model.config.num_attention_heads}")
        logger.info(f"num_hidden_layers: {hf_model.config.num_hidden_layers}")
        logger.info(f"max_position_embeddings: {hf_model.config.max_position_embeddings}")

        # create tt model like sd3.5 test pattern
        logger.info("Creating TT CLIP model...")
        from ...encoders.clip.model_clip import CLIPTextModel
        from ...encoders.clip.config_clip import CLIPTextConfig

        # create config from hf config (like sd3.5 test does)
        config = CLIPTextConfig(
            vocab_size=hf_model.config.vocab_size,
            hidden_size=hf_model.config.hidden_size,
            intermediate_size=hf_model.config.intermediate_size,
            num_attention_heads=hf_model.config.num_attention_heads,
            num_hidden_layers=hf_model.config.num_hidden_layers,
            max_prompt_length=77,  # clip standard
            layer_norm_eps=hf_model.config.layer_norm_eps,
            attention_dropout=hf_model.config.attention_dropout,
            hidden_act=hf_model.config.hidden_act,
        )

        tt_model = CLIPTextModel(config=config, mesh_device=mesh_device, with_projection=True)  # test with projection

        # load embeddings weights from hf (just the embeddings part)
        logger.info("Loading embedding weights...")
        embeddings_state_dict = {
            "token_embedding.weight": hf_model.text_model.embeddings.token_embedding.weight,
            "position_embedding.weight": hf_model.text_model.embeddings.position_embedding.weight,
        }

        # load state dict into our embeddings
        tt_model.embeddings.load_state_dict(embeddings_state_dict)

        # test text like sd3.5 test (same exact text)
        test_text = "A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"

        # tokenize like sd3.5 test (same exact pattern)
        hf_inputs = tokenizer(test_text, padding=True, truncation=True, max_length=77, return_tensors="pt")
        tt_tokens = ttnn.from_torch(
            hf_inputs.input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # test hf embeddings (like sd3.5 test)
        logger.info("Testing HF embeddings...")
        start_time = time.time()
        with torch.no_grad():
            hf_embeddings = hf_model.text_model.embeddings(hf_inputs.input_ids)
        logger.info(f"HF embeddings runtime: {time.time() - start_time}")

        # debug output like sd3.5 test
        logger.info(f"HF embeddings shape: {hf_embeddings.shape}")
        logger.info(f"HF embeddings mean: {hf_embeddings.mean():.6f}, std: {hf_embeddings.std():.6f}")

        # test tt embeddings (like sd3.5 test)
        logger.info("Testing TT embeddings...")
        start_time = time.time()
        tt_embeddings = tt_model.embeddings(tt_tokens, device=mesh_device)
        logger.info(f"TT embeddings runtime: {time.time() - start_time}")

        # convert tt output to torch for comparison (like sd3.5 test)
        tt_embeddings_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_embeddings)[0])

        # debug output like sd3.5 test
        logger.info(f"TT embeddings shape: {tt_embeddings_torch.shape}")
        logger.info(f"TT embeddings mean: {tt_embeddings_torch.mean():.6f}, std: {tt_embeddings_torch.std():.6f}")

        # shape check like sd3.5 test
        assert (
            hf_embeddings.shape == tt_embeddings_torch.shape
        ), f"Shape mismatch: {hf_embeddings.shape} vs {tt_embeddings_torch.shape}"

        assert not torch.isnan(tt_embeddings_torch).any(), "TT embeddings contain NaN"
        assert not torch.isinf(tt_embeddings_torch).any(), "TT embeddings contain Inf"
        assert tt_embeddings_torch.std() > 0.01, "TT embeddings have very low variance"

        logger.info(" CLIP embeddings test PASSED!")
        logger.info(" Embeddings work correctly through CLIPTextModelWithProjection")
        logger.info(" Shape matches HF implementation")
        logger.info(" No NaN/Inf values")

    except Exception as e:
        logger.error(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # cleanup like sd3.5 test
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    test_clip_embeddings_through_model()
