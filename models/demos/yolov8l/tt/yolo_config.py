import json

from ultralytics import YOLO


def extract_yolov8l_config(model_name="yolov8l.pt"):
    """Extract configuration parameters from YOLOv8L model"""

    model = YOLO(model_name).model
    model.eval()

    config = {
        "conv_config": {"input_params": []},
        "sppf_configs": {"input_params": []},
        "c2f_configs": {},
        "detect_config": {"cv2_params": [], "cv3_params": [], "dfl_params": {"input_params": []}},
    }

    # Track layer info
    layer_info = {}
    for name, module in model.named_modules():
        if hasattr(module, "conv") or hasattr(module, "weight"):
            if hasattr(module, "conv"):
                conv = module.conv
            else:
                conv = module

            if hasattr(conv, "kernel_size"):
                layer_info[name] = {
                    "kernel_size": conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size,
                    "stride": conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride,
                    "padding": conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
                    "in_channels": conv.in_channels,
                    "out_channels": conv.out_channels,
                }

    # Extract standalone conv layers (model.0, model.1, etc.)
    for i in range(20):  # Check up to model.19
        layer_name = f"model.{i}"
        if layer_name in layer_info:
            info = layer_info[layer_name]
            config["conv_config"]["input_params"].append(
                [info["kernel_size"], info["stride"], info["padding"], info["out_channels"], info["in_channels"]]
            )

    # Extract SPPF parameters (Ultralytics registers cv1/cv2 under model.9, not model.9 as a Conv)
    cv1_name = "model.9.cv1"
    cv2_name = "model.9.cv2"
    if cv1_name in layer_info and cv2_name in layer_info:
        cv1_info = layer_info[cv1_name]
        cv2_info = layer_info[cv2_name]
        config["sppf_configs"]["input_params"].append([1, 1, 0, cv1_info["out_channels"], cv1_info["in_channels"]])
        config["sppf_configs"]["input_params"].append([1, 1, 0, cv2_info["out_channels"], cv2_info["in_channels"]])

    # Extract C2f parameters
    c2f_modules = ["2", "4", "6", "8", "12", "15", "18", "21"]
    for module_num in c2f_modules:
        module_name = f"model.{module_num}"

        # Get cv1 info
        cv1_name = f"{module_name}.cv1"
        cv1_info = layer_info.get(cv1_name, {})

        # Get cv2 info
        cv2_name = f"{module_name}.cv2"
        cv2_info = layer_info.get(cv2_name, {})

        # Get bottleneck info (first bottleneck)
        bott_cv1_name = f"{module_name}.m.0.cv1"
        bott_cv1_info = layer_info.get(bott_cv1_name, {})

        bott_cv2_name = f"{module_name}.m.0.cv2"
        bott_cv2_info = layer_info.get(bott_cv2_name, {})

        if cv1_info and cv2_info and bott_cv1_info and bott_cv2_info:
            config["c2f_configs"][module_name] = {
                "input_params": [
                    [1, 1, 0, cv1_info["out_channels"], cv1_info["in_channels"]],  # cv1
                    [1, 1, 0, cv2_info["out_channels"], cv2_info["in_channels"]],  # cv2
                    [3, 1, 1, bott_cv1_info["out_channels"], bott_cv1_info["in_channels"]],  # bott cv1
                    [1, 1, 0, bott_cv2_info["out_channels"], bott_cv2_info["in_channels"]],  # bott cv2
                    [1, 1, 0, bott_cv2_info["out_channels"], bott_cv2_info["in_channels"]],  # duplicate for split
                ]
            }

    # Extract detection head parameters
    # cv2_params (3 levels)
    for i in range(3):
        cv2_0 = layer_info.get(f"model.22.cv2.{i}.0", {})
        cv2_1 = layer_info.get(f"model.22.cv2.{i}.1", {})
        cv2_2 = layer_info.get(f"model.22.cv2.{i}.2", {})

        if cv2_0 and cv2_1 and cv2_2:
            config["detect_config"]["cv2_params"].append(
                {
                    "input_params": [
                        [3, 1, 1, cv2_0["out_channels"], cv2_0["in_channels"]],
                        [3, 1, 1, cv2_1["out_channels"], cv2_1["in_channels"]],
                        [1, 1, 0, cv2_2["out_channels"], cv2_2["in_channels"]],
                    ]
                }
            )

    # cv3_params (3 levels)
    for i in range(3):
        cv3_0 = layer_info.get(f"model.22.cv3.{i}.0", {})
        cv3_1 = layer_info.get(f"model.22.cv3.{i}.1", {})
        cv3_2 = layer_info.get(f"model.22.cv3.{i}.2", {})

        if cv3_0 and cv3_1 and cv3_2:
            config["detect_config"]["cv3_params"].append(
                {
                    "input_params": [
                        [3, 1, 1, cv3_0["out_channels"], cv3_0["in_channels"]],
                        [3, 1, 1, cv3_1["out_channels"], cv3_1["in_channels"]],
                        [1, 1, 0, cv3_2["out_channels"], cv3_2["in_channels"]],
                    ]
                }
            )

    # DFL parameters
    dfl_info = layer_info.get("model.22.dfl", {})
    if dfl_info:
        config["detect_config"]["dfl_params"]["input_params"] = [
            1,
            1,
            0,
            dfl_info["out_channels"],
            dfl_info["in_channels"],
        ]

    return config


def generate_yolov8l_config():
    """Generate and save YOLOv8L config file"""

    config = extract_yolov8l_config("yolov8l.pt")

    # Print formatted JSON
    print(json.dumps(config, indent=4))

    # Save to file
    with open("yolov8l_config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nConfig saved to yolov8l_config.json")


if __name__ == "__main__":
    generate_yolov8l_config()
