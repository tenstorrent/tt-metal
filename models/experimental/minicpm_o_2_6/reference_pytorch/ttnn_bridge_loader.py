#!/usr/bin/env python3
"""
TTNN Bridge Loader - Use TTNN's efficient loading infrastructure for PyTorch models
"""

import sys
from pathlib import Path
from typing import Dict, Any
import torch
from loguru import logger

# Add TTNN paths - try multiple potential locations
current_dir = Path(__file__).parent
possible_ttnn_paths = [
    current_dir.parent.parent.parent.parent / "models/tt_transformers/tt",  # From minicpm_o_2_6
    current_dir.parent.parent.parent / "models/tt_transformers/tt",  # From experimental
    Path("/home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal/models/tt_transformers/tt"),
]

ttnn_path = None
for path in possible_ttnn_paths:
    if path.exists():
        ttnn_path = path
        break

if ttnn_path and str(ttnn_path) not in sys.path:
    sys.path.insert(0, str(ttnn_path))
    logger.info(f"✅ Added TTNN path: {ttnn_path}")
else:
    logger.warning("⚠️ TTNN path not found in expected locations")


class TTNNBridgeLoader:
    """Bridge loader that uses TTNN's infrastructure to load large PyTorch models efficiently"""

    def __init__(self, model_dir: str = "model_cache/minicpm_o_2_6_int4"):
        self.model_name = "openbmb/MiniCPM-o-2_6-int4"  # Use GPTQ quantized version
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Import TTNN's loading functions
        try:
            from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict

            self.load_hf_state_dict = load_hf_state_dict
            logger.info("✅ TTNN loading infrastructure available")
        except ImportError as e:
            raise ImportError(f"TTNN loading infrastructure not available: {e}")

    def load_full_model(self) -> Dict[str, torch.Tensor]:
        """Load the complete model using TTNN's efficient loading"""
        logger.info(f"🚀 Loading {self.model_name} (GPTQ quantized) using TTNN infrastructure...")

        try:
            # Ensure model is downloaded first
            self._ensure_model_downloaded()

            # Use TTNN's efficient loading
            state_dict = self.load_hf_state_dict(str(self.model_dir))
            total_params = sum(tensor.numel() for tensor in state_dict.values())

            # For GPTQ models, calculate effective memory (4-bit quantization)
            memory_gb = total_params * 0.5 / (1024**3)  # ~0.5 bytes per param for 4-bit

            logger.info(f"📦 Loaded {len(state_dict)} tensors efficiently")
            logger.info(f"📊 Effective memory: {memory_gb:.2f} GB (4-bit quantized)")

            return state_dict

        except Exception as e:
            logger.error(f"❌ Failed to load model with TTNN infrastructure: {e}")
            raise

    def _ensure_model_downloaded(self):
        """Ensure the GPTQ model is downloaded"""
        from huggingface_hub import snapshot_download

        if not any(self.model_dir.glob("*.safetensors")):
            logger.info(f"📥 Downloading {self.model_name}...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False,
                token=None,  # Add token if needed for private models
            )
            logger.info("✅ Model downloaded successfully")
        else:
            logger.info("✅ Model already downloaded")

    def load_component(self, component_prefix: str) -> Dict[str, torch.Tensor]:
        """Load only a specific component (e.g., 'llm.', 'vpm.', 'apm.')"""
        logger.info(f"📦 Loading component: {component_prefix}")

        full_state_dict = self.load_full_model()

        # Filter for component
        component_dict = {k: v for k, v in full_state_dict.items() if k.startswith(component_prefix)}

        if not component_dict:
            logger.warning(f"⚠️ No tensors found for component: {component_prefix}")
            return {}

        logger.info(f"✅ Loaded {len(component_dict)} tensors for {component_prefix}")
        return component_dict

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded model"""
        try:
            state_dict = self.load_full_model()

            # Count parameters by component
            component_counts = {}
            total_params = 0

            for key, tensor in state_dict.items():
                params = tensor.numel()
                total_params += params

                # Determine component
                if key.startswith("llm."):
                    comp = "llm"
                elif key.startswith("vpm."):
                    comp = "vision"
                elif key.startswith("apm."):
                    comp = "audio"
                elif key.startswith("resampler."):
                    comp = "resampler"
                elif key.startswith("audio_projection_layer."):
                    comp = "audio_proj"
                elif key.startswith("tts."):
                    comp = "tts"
                else:
                    comp = "other"

                component_counts[comp] = component_counts.get(comp, 0) + params

            memory_gb = total_params * 4 / (1024**3)

            return {
                "total_parameters": total_params,
                "memory_gb": memory_gb,
                "num_tensors": len(state_dict),
                "component_breakdown": component_counts,
            }

        except Exception as e:
            logger.error(f"❌ Failed to get model stats: {e}")
            return {}


def create_ttnn_bridge_loader(model_dir: str = "model_cache/minicpm_o_2_6") -> TTNNBridgeLoader:
    """Factory function to create TTNN bridge loader"""
    return TTNNBridgeLoader(model_dir)


if __name__ == "__main__":
    # Test the bridge loader
    import argparse

    parser = argparse.ArgumentParser(description="TTNN Bridge Loader Test")
    parser.add_argument("--model_dir", default="model_cache/minicpm_o_2_6", help="Model directory")
    parser.add_argument("--component", help="Load specific component (e.g., 'llm.', 'vpm.')")
    parser.add_argument("--stats", action="store_true", help="Show model statistics")

    args = parser.parse_args()

    try:
        loader = create_ttnn_bridge_loader(args.model_dir)

        if args.stats:
            stats = loader.get_model_stats()
            print("📊 Model Statistics:")
            print(f"   Total Parameters: {stats.get('total_parameters', 0):,}")
            print(f"   Number of Tensors: {stats.get('num_tensors', 0)}")

            print("   Component Breakdown:")
            for comp, params in stats.get("component_breakdown", {}).items():
                print(f"     {comp}: {params:,} parameters")

        elif args.component:
            component_dict = loader.load_component(args.component)
            print(f"✅ Loaded {len(component_dict)} tensors for {args.component}")

        else:
            # Load full model
            state_dict = loader.load_full_model()
            print(f"✅ Successfully loaded full model with {len(state_dict)} tensors")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
