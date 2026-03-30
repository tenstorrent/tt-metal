# import torch
# from ultralytics import YOLO
# import re

# def find_conv_bn_pairs_and_c2f_paths(model_name='yolov8l.pt'):
#     """Extract conv+bn pairs and C2F paths from YOLOv8 model"""

#     # Load YOLOv8 model
#     model = YOLO(model_name).model
#     model.eval()

#     pairs = []
#     c2f_paths = []

#     # First pass: identify all modules and their types
#     module_info = {}
#     for name, module in model.named_modules():
#         if '.' in name:  # Skip root module
#             module_info[name] = type(module).__name__

#     # Second pass: find conv+bn pairs
#     for name, module_type in module_info.items():
#         # Check for standalone conv layers (model.0, model.1, etc.)
#         if (module_type == 'Conv' and
#             not any(x in name for x in ['cv1', 'cv2', 'm.', 'cv3', 'dfl']) and
#             re.match(r'model\.\d+$', name)):  # Match model.0, model.1, etc.

#             pairs.append((name, True))  # True for bfloat8

#     # Find C2F modules and their internal conv+bn pairs
#     c2f_modules = [name for name, mtype in module_info.items()
#                    if 'C2f' in mtype and '.' in name]

#     for c2f_name in c2f_modules:
#         # Add cv1 and cv2 paths
#         cv1_path = f"{c2f_name}.cv1"
#         cv2_path = f"{c2f_name}.cv2"

#         if cv1_path in module_info:
#             c2f_paths.append(cv1_path)
#             pairs.append((cv1_path, True))

#         if cv2_path in module_info:
#             pairs.append((cv2_path, True))

#         # Find bottleneck modules (m.*) within this C2F
#         bottleneck_paths = [name for name in module_info.keys()
#                            if name.startswith(f"{c2f_name}.m.") and
#                            name.endswith('.cv1')]

#         for bottleneck_path in bottleneck_paths:
#             # Add cv1 and cv2 for each bottleneck
#             base_path = bottleneck_path[:-4]  # Remove '.cv1'
#             cv1_path = f"{base_path}.cv1"
#             cv2_path = f"{base_path}.cv2"

#             if cv1_path in module_info:
#                 pairs.append((cv1_path, True))
#             if cv2_path in module_info:
#                 pairs.append((cv2_path, True))

#     # Find SPPF module conv layers
#     sppf_modules = [name for name, mtype in module_info.items()
#                     if 'SPPF' in mtype and '.' in name]

#     for sppf_name in sppf_modules:
#         cv1_path = f"{sppf_name}.cv1"
#         cv2_path = f"{sppf_name}.cv2"

#         if cv1_path in module_info:
#             pairs.append((cv1_path, True))
#         if cv2_path in module_info:
#             pairs.append((cv2_path, True))

#     # Find detection head paths - only the inner conv layers
#     detect_modules = [name for name in module_info.keys()
#                       if name.startswith('model.22.') and
#                       (name.endswith('.0') or name.endswith('.1'))]

#     for path in sorted(detect_modules):
#         # Only include the innermost conv layers (like .0.0, .0.1)
#         if re.search(r'\.\d+\.\d+$', path):
#             pairs.append((path, True))

#     # Sort pairs by module hierarchy for consistency
#     pairs.sort(key=lambda x: x[0])

#     return pairs, c2f_paths

# def print_yolov8_structure():
#     """Print the complete structure for YOLOv8"""

#     pairs, c2f_paths = find_conv_bn_pairs_and_c2f_paths()

#     print("pairs = [")
#     for path, bfloat8 in pairs:
#         print(f'    ("{path}", {bfloat8}),')
#     print("]")

#     print("\nc2f_paths = [")
#     for path in c2f_paths:
#         print(f'    "{path}",')
#     print("]")

# if __name__ == "__main__":
#     print_yolov8_structure()


from ultralytics import YOLO

model = YOLO("yolov8l.pt").model
model.eval()

# Check detection head structure
for name, module in model.named_modules():
    if "model.22" in name and any(x in name for x in ["cv2", "cv3"]):
        print(f"{name}: {type(module).__name__}")
