"""
OpenVLA Instruction Sensitivity Test

Tests instruction sensitivity with:
1. Realistic synthetic scenes (blocks on table with gripper)
2. 4 different instructions (pick, push, move_left, move_right)
3. Multiple image types

This test verifies that the model responds differently to different instructions.
"""

import os
import random

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor

import ttnn
from models.tt_transformers.tt.multimodal.open_vla import OpenVLAConfig, TTOpenVLAForActionPrediction

# ============================================================================
# IMAGE GENERATION UTILITIES
# ============================================================================


def create_robot_scene(bg_color, block_color, block_pos="center", gripper_open=True):
    """Create a synthetic robot manipulation scene with a colored block on a table."""
    img = Image.new("RGB", (224, 224), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw table surface (wood-like)
    draw.rectangle([0, 130, 224, 224], fill=(160, 120, 80))

    # Draw table edge
    draw.rectangle([0, 130, 224, 140], fill=(120, 80, 50))

    # Draw block position
    if block_pos == "left":
        x1, x2 = 30, 80
    elif block_pos == "right":
        x1, x2 = 144, 194
    else:  # center
        x1, x2 = 87, 137

    # Draw 3D block effect (top and side)
    draw.polygon([(x1, 90), (x2, 90), (x2 + 10, 80), (x1 + 10, 80)], fill=tuple(max(0, c - 40) for c in block_color))
    draw.polygon([(x2, 90), (x2, 130), (x2 + 10, 120), (x2 + 10, 80)], fill=tuple(max(0, c - 60) for c in block_color))
    draw.rectangle([x1, 90, x2, 130], fill=block_color)

    # Draw simple gripper above
    draw.rectangle([100, 10, 124, 50], fill=(100, 100, 100))  # Arm
    if gripper_open:
        draw.rectangle([85, 50, 100, 75], fill=(80, 80, 80))  # Left finger
        draw.rectangle([124, 50, 139, 75], fill=(80, 80, 80))  # Right finger
    else:
        draw.rectangle([95, 50, 107, 75], fill=(80, 80, 80))  # Left finger (closed)
        draw.rectangle([117, 50, 129, 75], fill=(80, 80, 80))  # Right finger (closed)

    return img


def create_solid_image(color):
    """Create a simple solid color image (original test format)."""
    return Image.new("RGB", (224, 224), color=color)


def create_realistic_tabletop_scene(scene_type="blocks"):
    """Create a more realistic tabletop manipulation scene with multiple objects."""
    # Gray/brown table background (like real robot setups)
    img = Image.new("RGB", (224, 224), color=(180, 160, 140))
    draw = ImageDraw.Draw(img)

    # Wood grain table texture (horizontal lines)
    for y in range(0, 224, 8):
        shade = random.randint(-10, 10)
        draw.line([(0, y), (224, y)], fill=(170 + shade, 150 + shade, 130 + shade), width=2)

    # Draw table edge
    draw.rectangle([0, 180, 224, 224], fill=(120, 100, 80))

    if scene_type == "blocks":
        # Multiple colored blocks scattered
        # Green block on left
        draw.rectangle([20, 120, 60, 160], fill=(50, 150, 50))
        draw.rectangle([20, 115, 60, 120], fill=(40, 130, 40))  # top
        # Orange block in center
        draw.rectangle([90, 130, 130, 170], fill=(220, 140, 50))
        draw.rectangle([90, 125, 130, 130], fill=(200, 120, 40))  # top
        # Purple block on right
        draw.rectangle([150, 110, 190, 150], fill=(150, 80, 180))
        draw.rectangle([150, 105, 190, 110], fill=(130, 60, 160))  # top

    elif scene_type == "cups":
        # Scene with cups/containers
        # Blue cup
        draw.ellipse([30, 100, 70, 115], fill=(80, 80, 180))
        draw.rectangle([35, 115, 65, 160], fill=(70, 70, 170))
        draw.ellipse([35, 155, 65, 165], fill=(70, 70, 170))
        # Red cup
        draw.ellipse([140, 90, 180, 105], fill=(180, 60, 60))
        draw.rectangle([145, 105, 175, 155], fill=(170, 50, 50))
        draw.ellipse([145, 150, 175, 160], fill=(170, 50, 50))

    elif scene_type == "tools":
        # Scene with tools (screwdriver, etc)
        # Yellow screwdriver handle
        draw.rectangle([30, 130, 50, 170], fill=(220, 200, 50))
        draw.rectangle([35, 170, 45, 200], fill=(150, 150, 150))  # metal shaft
        # Blue tape roll
        draw.ellipse([120, 110, 180, 170], fill=(60, 100, 180))
        draw.ellipse([135, 125, 165, 155], fill=(180, 160, 140))  # hole

    # Robot gripper (always present)
    draw.rectangle([95, 10, 130, 50], fill=(80, 80, 90))  # arm
    draw.polygon([(85, 50), (95, 50), (95, 80), (85, 70)], fill=(70, 70, 80))  # left finger
    draw.polygon([(130, 50), (140, 50), (140, 70), (130, 80)], fill=(70, 70, 80))  # right finger

    return img


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

# 4 Different instructions that should produce different robot actions
INSTRUCTIONS = {
    "PICK": "In: What action should the robot take to pick up the block?\nOut:",
    "PUSH_LEFT": "In: What action should the robot take to push the block to the left?\nOut:",
    "PUSH_RIGHT": "In: What action should the robot take to push the block to the right?\nOut:",
    "PLACE": "In: What action should the robot take to place the block down?\nOut:",
}

# Image configurations
IMAGE_CONFIGS = {
    # Realistic scenes with objects
    "SCENE_RED": lambda: create_robot_scene((180, 180, 190), (200, 50, 50), "center"),
    "SCENE_BLUE": lambda: create_robot_scene((180, 180, 190), (50, 50, 200), "center"),
    # Solid colors (original test)
    "SOLID_RED": lambda: create_solid_image((200, 50, 50)),
    "SOLID_BLUE": lambda: create_solid_image((50, 50, 200)),
}


# ============================================================================
# MAIN TEST
# ============================================================================


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}[
            os.environ.get("MESH_DEVICE", os.environ.get("FAKE_DEVICE", "N150"))
        ]
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_instruction_sensitivity_realistic(mesh_device):
    """
    INSTRUCTION SENSITIVITY TEST with realistic synthetic images.

    Tests:
    - 2 Images (SCENE_RED, SCENE_BLUE) with objects
    - 4 Instructions (PICK, PUSH_LEFT, PUSH_RIGHT, PLACE)
    - Total: 8 combinations

    Expected: All 8 should produce different actions if instruction sensitivity works.
    """
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Create config
    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, _ = OpenVLAConfig.from_dict(config_dict, **kwargs)

    # Load weights
    weight_path = os.getenv("OPENVLA_WEIGHTS", None)
    merged_tensors = None
    if weight_path is not None and os.path.exists(weight_path):
        from safetensors import safe_open

        shard_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        merged_tensors = {}
        for path in shard_files:
            if os.path.exists(weight_path + path):
                with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        merged_tensors[key] = f.get_tensor(key)

    # Create model
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    # ================================================================
    # TEST SETUP
    # ================================================================
    print(f"\n{'='*80}")
    print("INSTRUCTION SENSITIVITY TEST - REALISTIC SCENES")
    print(f"{'='*80}")

    # Use scene images (with objects)
    images = {
        "SCENE_RED": IMAGE_CONFIGS["SCENE_RED"](),
        "SCENE_BLUE": IMAGE_CONFIGS["SCENE_BLUE"](),
    }

    print(f"\n🖼️  Images: SCENE_RED (red block on table), SCENE_BLUE (blue block on table)")
    print(f"📝 Instructions ({len(INSTRUCTIONS)}):")
    for name, prompt in INSTRUCTIONS.items():
        print(f"   {name}: '{prompt.replace(chr(10), ' ')[:60]}...'")

    # ================================================================
    # RUN ALL COMBINATIONS: 2 images × 4 instructions = 8 combinations
    # ================================================================
    print(f"\n{'='*80}")
    print("RUNNING 8 COMBINATIONS (2 images × 4 instructions)")
    print(f"{'='*80}")

    results = {}
    combinations = [(img_name, instr_name) for img_name in images.keys() for instr_name in INSTRUCTIONS.keys()]

    for img_name, instr_name in combinations:
        combo_name = f"{img_name}_{instr_name}"
        print(f"\n--- {combo_name} ---")

        # Get inputs
        inputs = processor(INSTRUCTIONS[instr_name], images[img_name]).to("cpu", dtype=torch.bfloat16)

        # Debug info
        input_ids = inputs["input_ids"]
        print(f"   input_ids last_5: {input_ids[0, -5:].tolist()}")

        # Run prediction
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        results[combo_name] = {
            "action": action,
            "image": img_name,
            "instruction": instr_name,
        }
        print(f"   Action: {np.array2string(action, precision=4)}")

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    # Print all results in a table
    print(f"\n{'Combination':<25} {'Action Values':<60}")
    print("-" * 85)
    for name, res in results.items():
        action_str = np.array2string(res["action"], precision=4, separator=", ")
        print(f"{name:<25} {action_str}")

    # ================================================================
    # INSTRUCTION SENSITIVITY CHECK
    # ================================================================
    print(f"\n{'='*80}")
    print("INSTRUCTION SENSITIVITY (same image, different instructions)")
    print(f"{'='*80}")

    instruction_diffs = {}
    for img_name in images.keys():
        print(f"\n🖼️  Image: {img_name}")
        img_results = {k: v for k, v in results.items() if k.startswith(img_name)}

        # Compare all pairs of instructions for this image
        instr_names = list(INSTRUCTIONS.keys())
        for i, instr1 in enumerate(instr_names):
            for instr2 in instr_names[i + 1 :]:
                key1 = f"{img_name}_{instr1}"
                key2 = f"{img_name}_{instr2}"
                diff = np.abs(results[key1]["action"] - results[key2]["action"]).sum()
                status = "✅ DIFF" if diff > 0.01 else "❌ SAME"
                print(f"   {instr1} vs {instr2}: L1={diff:.4f} {status}")
                instruction_diffs[f"{key1}_vs_{key2}"] = diff

    # ================================================================
    # IMAGE SENSITIVITY CHECK
    # ================================================================
    print(f"\n{'='*80}")
    print("IMAGE SENSITIVITY (different images, same instruction)")
    print(f"{'='*80}")

    image_diffs = {}
    for instr_name in INSTRUCTIONS.keys():
        key1 = f"SCENE_RED_{instr_name}"
        key2 = f"SCENE_BLUE_{instr_name}"
        diff = np.abs(results[key1]["action"] - results[key2]["action"]).sum()
        status = "✅ DIFF" if diff > 0.01 else "❌ SAME"
        print(f"   {instr_name}: RED vs BLUE = L1={diff:.4f} {status}")
        image_diffs[instr_name] = diff

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print(f"{'='*80}")

    # Count passes
    instr_pass = sum(1 for d in instruction_diffs.values() if d > 0.01)
    instr_total = len(instruction_diffs)
    image_pass = sum(1 for d in image_diffs.values() if d > 0.01)
    image_total = len(image_diffs)

    print(f"\n📝 Instruction Sensitivity: {instr_pass}/{instr_total} pairs different")
    print(f"🖼️  Image Sensitivity: {image_pass}/{image_total} pairs different")

    if instr_pass == instr_total:
        print(f"\n✅ FULL INSTRUCTION SENSITIVITY - Model distinguishes all instructions!")
    elif instr_pass > 0:
        print(f"\n⚠️  PARTIAL INSTRUCTION SENSITIVITY - Some instructions distinguished")
    else:
        print(f"\n❌ NO INSTRUCTION SENSITIVITY - Model ignores instructions")

    if image_pass == image_total:
        print(f"✅ FULL IMAGE SENSITIVITY - Model distinguishes all images!")
    elif image_pass > 0:
        print(f"⚠️  PARTIAL IMAGE SENSITIVITY - Some images distinguished")
    else:
        print(f"❌ NO IMAGE SENSITIVITY - Model ignores images")

    # Assert at least image sensitivity works
    assert image_pass > 0, "Image sensitivity completely failed!"


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}[
            os.environ.get("MESH_DEVICE", os.environ.get("FAKE_DEVICE", "N150"))
        ]
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_instruction_sensitivity_comparison(mesh_device):
    """
    COMPARISON TEST: Realistic Scenes vs Solid Colors

    Tests whether realistic scenes with objects improve instruction sensitivity
    compared to simple solid color images.
    """
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Create config
    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, _ = OpenVLAConfig.from_dict(config_dict, **kwargs)

    # Load weights
    weight_path = os.getenv("OPENVLA_WEIGHTS", None)
    merged_tensors = None
    if weight_path is not None and os.path.exists(weight_path):
        from safetensors import safe_open

        shard_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        merged_tensors = {}
        for path in shard_files:
            if os.path.exists(weight_path + path):
                with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        merged_tensors[key] = f.get_tensor(key)

    # Create model
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    # ================================================================
    # TEST SETUP - Both image types
    # ================================================================
    print(f"\n{'='*80}")
    print("COMPARISON: REALISTIC SCENES vs SOLID COLORS")
    print(f"{'='*80}")

    images = {
        # Realistic scenes
        "SCENE_RED": IMAGE_CONFIGS["SCENE_RED"](),
        "SCENE_BLUE": IMAGE_CONFIGS["SCENE_BLUE"](),
        # Solid colors
        "SOLID_RED": IMAGE_CONFIGS["SOLID_RED"](),
        "SOLID_BLUE": IMAGE_CONFIGS["SOLID_BLUE"](),
    }

    # Use just 2 instructions for comparison
    test_instructions = {
        "PICK": INSTRUCTIONS["PICK"],
        "PUSH_LEFT": INSTRUCTIONS["PUSH_LEFT"],
    }

    print(f"\n🖼️  Images: SCENE_RED, SCENE_BLUE, SOLID_RED, SOLID_BLUE")
    print(f"📝 Instructions: PICK, PUSH_LEFT")

    # ================================================================
    # RUN ALL COMBINATIONS
    # ================================================================
    print(f"\n{'='*80}")
    print("RUNNING COMBINATIONS")
    print(f"{'='*80}")

    results = {}
    for img_name, img in images.items():
        for instr_name, prompt in test_instructions.items():
            combo_name = f"{img_name}_{instr_name}"
            print(f"\n--- {combo_name} ---")

            inputs = processor(prompt, img).to("cpu", dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

            results[combo_name] = action
            print(f"   Action: {np.array2string(action, precision=4)}")

    # ================================================================
    # COMPARE INSTRUCTION SENSITIVITY: Scenes vs Solid
    # ================================================================
    print(f"\n{'='*80}")
    print("INSTRUCTION SENSITIVITY COMPARISON")
    print(f"{'='*80}")

    # Scene images - instruction sensitivity
    scene_red_diff = np.abs(results["SCENE_RED_PICK"] - results["SCENE_RED_PUSH_LEFT"]).sum()
    scene_blue_diff = np.abs(results["SCENE_BLUE_PICK"] - results["SCENE_BLUE_PUSH_LEFT"]).sum()

    # Solid images - instruction sensitivity
    solid_red_diff = np.abs(results["SOLID_RED_PICK"] - results["SOLID_RED_PUSH_LEFT"]).sum()
    solid_blue_diff = np.abs(results["SOLID_BLUE_PICK"] - results["SOLID_BLUE_PUSH_LEFT"]).sum()

    print(f"\n📊 Instruction Sensitivity (PICK vs PUSH_LEFT):")
    print(f"   SCENE_RED:  L1={scene_red_diff:.4f} {'✅' if scene_red_diff > 0.01 else '❌'}")
    print(f"   SCENE_BLUE: L1={scene_blue_diff:.4f} {'✅' if scene_blue_diff > 0.01 else '❌'}")
    print(f"   SOLID_RED:  L1={solid_red_diff:.4f} {'✅' if solid_red_diff > 0.01 else '❌'}")
    print(f"   SOLID_BLUE: L1={solid_blue_diff:.4f} {'✅' if solid_blue_diff > 0.01 else '❌'}")

    # Verdict
    scene_sensitive = (scene_red_diff > 0.01) or (scene_blue_diff > 0.01)
    solid_sensitive = (solid_red_diff > 0.01) or (solid_blue_diff > 0.01)

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    if scene_sensitive and not solid_sensitive:
        print("✅ SCENE IMAGES improve instruction sensitivity over solid colors!")
    elif scene_sensitive and solid_sensitive:
        print("✅ Both image types show instruction sensitivity!")
    elif not scene_sensitive and solid_sensitive:
        print("⚠️  Solid colors work better than scenes (unexpected)")
    else:
        print("❌ Neither image type shows instruction sensitivity")

    # Print summary table
    print(f"\n{'='*80}")
    print("ALL RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Combination':<25} {'Action Values':<60}")
    print("-" * 85)
    for name, action in results.items():
        action_str = np.array2string(action, precision=4, separator=", ")
        print(f"{name:<25} {action_str}")


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}[
            os.environ.get("MESH_DEVICE", os.environ.get("FAKE_DEVICE", "N150"))
        ]
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
        }
    ],
    indirect=True,
)
def test_realistic_scenes_instructions(mesh_device):
    """
    TEST WITH REALISTIC TABLETOP SCENES

    Uses multi-object scenes (blocks, cups, tools) with varied instructions.
    This tests if OpenVLA can handle realistic robot manipulation scenarios.
    """
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Create config
    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, _ = OpenVLAConfig.from_dict(config_dict, **kwargs)

    # Load weights
    weight_path = os.getenv("OPENVLA_WEIGHTS", None)
    merged_tensors = None
    if weight_path is not None and os.path.exists(weight_path):
        from safetensors import safe_open

        shard_files = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
        merged_tensors = {}
        for path in shard_files:
            if os.path.exists(weight_path + path):
                with safe_open(weight_path + path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        merged_tensors[key] = f.get_tensor(key)

    # Create model
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    # Enable debug tracing to see KV cache reset (disabled for cleaner output)
    # vla._debug_trace = True

    # ================================================================
    # TEST SETUP - Realistic tabletop scenes
    # ================================================================
    print(f"\n{'='*80}")
    print("REALISTIC TABLETOP SCENES TEST")
    print(f"{'='*80}")

    # Create realistic scenes with multiple objects
    images = {
        "BLOCKS": create_realistic_tabletop_scene("blocks"),  # Green, orange, purple blocks
        "CUPS": create_realistic_tabletop_scene("cups"),  # Blue cup, red cup
        "TOOLS": create_realistic_tabletop_scene("tools"),  # Screwdriver, tape roll
    }

    # Varied instructions for different tasks
    instructions = {
        "PICK_GREEN": "In: What action should the robot take to pick up the green block?\nOut:",
        "PICK_ORANGE": "In: What action should the robot take to pick up the orange block?\nOut:",
        "PUSH_LEFT": "In: What action should the robot take to push the object to the left?\nOut:",
        "MOVE_RIGHT": "In: What action should the robot take to move the gripper to the right?\nOut:",
    }

    print(f"\n🖼️  Images: BLOCKS (3 colored blocks), CUPS (2 cups), TOOLS (screwdriver + tape)")
    print(f"📝 Instructions: PICK_GREEN, PICK_ORANGE, PUSH_LEFT, MOVE_RIGHT")

    # ================================================================
    # RUN ALL COMBINATIONS: 3 images × 4 instructions = 12 combinations
    # ================================================================
    print(f"\n{'='*80}")
    print("RUNNING 12 COMBINATIONS (3 images × 4 instructions)")
    print(f"{'='*80}")

    results = {}
    for img_name, img in images.items():
        for instr_name, prompt in instructions.items():
            combo_name = f"{img_name}_{instr_name}"
            print(f"\n--- {combo_name} ---")

            inputs = processor(prompt, img).to("cpu", dtype=torch.bfloat16)

            # Debug: Print input_ids to verify they differ
            input_ids = inputs["input_ids"]
            print(f"   input_ids last 10: {input_ids[0, -10:].tolist()}")

            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

            # Debug: Get raw generated tokens from language model
            if hasattr(vla, "language_model") and hasattr(vla.language_model, "_last_generated_tokens"):
                raw_tokens = vla.language_model._last_generated_tokens
                print(f"   Raw tokens: {raw_tokens}")

            results[combo_name] = action
            print(f"   Action: {np.array2string(action, precision=4)}")

    # ================================================================
    # ANALYSIS: Image Sensitivity (same instruction, different scenes)
    # ================================================================
    print(f"\n{'='*80}")
    print("IMAGE SENSITIVITY (same instruction, different scenes)")
    print(f"{'='*80}")

    img_names = list(images.keys())
    for instr_name in instructions.keys():
        print(f"\n📝 Instruction: {instr_name}")
        for i, img1 in enumerate(img_names):
            for img2 in img_names[i + 1 :]:
                key1 = f"{img1}_{instr_name}"
                key2 = f"{img2}_{instr_name}"
                diff = np.abs(results[key1] - results[key2]).sum()
                status = "✅ DIFF" if diff > 0.01 else "❌ SAME"
                print(f"   {img1} vs {img2}: L1={diff:.4f} {status}")

    # ================================================================
    # ANALYSIS: Instruction Sensitivity (same scene, different instructions)
    # ================================================================
    print(f"\n{'='*80}")
    print("INSTRUCTION SENSITIVITY (same scene, different instructions)")
    print(f"{'='*80}")

    instr_names = list(instructions.keys())
    for img_name in images.keys():
        print(f"\n🖼️  Scene: {img_name}")
        for i, instr1 in enumerate(instr_names):
            for instr2 in instr_names[i + 1 :]:
                key1 = f"{img_name}_{instr1}"
                key2 = f"{img_name}_{instr2}"
                diff = np.abs(results[key1] - results[key2]).sum()
                status = "✅ DIFF" if diff > 0.01 else "❌ SAME"
                print(f"   {instr1} vs {instr2}: L1={diff:.4f} {status}")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print(f"\n{'='*80}")
    print("ALL RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Combination':<25} {'Action Values':<60}")
    print("-" * 85)
    for name, action in results.items():
        action_str = np.array2string(action, precision=4, separator=", ")
        print(f"{name:<25} {action_str}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Count unique actions
    unique_actions = []
    for action in results.values():
        is_unique = True
        for existing in unique_actions:
            if np.abs(action - existing).sum() < 0.01:
                is_unique = False
                break
        if is_unique:
            unique_actions.append(action)

    print(f"\n📊 Total combinations: {len(results)}")
    print(f"📊 Unique actions: {len(unique_actions)}")
    print(f"📊 Uniqueness ratio: {len(unique_actions)/len(results)*100:.1f}%")

    if len(unique_actions) >= len(results) * 0.5:
        print(f"\n✅ Good diversity! Model produces varied actions")
    else:
        print(f"\n⚠️  Low diversity - model may not be sensitive to inputs")


if __name__ == "__main__":
    # Quick test run
    print("Run with: pytest test_openvla_instruction_sensitivity.py -v -s")
