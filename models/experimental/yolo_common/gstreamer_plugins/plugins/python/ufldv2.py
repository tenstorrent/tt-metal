#!/usr/bin/env python3
import gi
import os
import time  # Import time module for FPS calculation & processing simulation
import numpy as np
import cv2
import torch

# os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase
import ttnn


# print("ARCH YAML   ", os.environ["WH_ARCH_YAML"])

from models.demos.ufld_v2.runner.performant_runner import UFLDPerformantRunner
from models.demos.ufld_v2.demo.demo_utils import (
    generate_tusimple_lines,
)

# from models.demos.ufld_v2.demo.demo_utils import load_coco_class_names
# from models.experimental.yolo_evaluation.yolo_evaluation_utils import postprocess as obj_postprocess


# Initialize GStreamer (only once)
Gst.init(None)


# --- Element Class Definition ---
class UFLDv2(GstBase.BaseTransform):
    # Element metadata (for GStreamer)

    __gtype_name__ = "GstUFLDv2PythonBatching"

    __gstmetadata__ = (
        "UFLDv2 Python",  # Long name
        "Filter/Effect/Converter",  # Classification
        "Prepends a configurable string to text buffer data",  # Description
        "Your Name <your.email@example.com>",  # Author
    )

    # Pad Templates
    _sink_template = Gst.PadTemplate.new(
        "sink",
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=BGRx,width=(int)800,height=(int)320,framerate=(fraction)30000/1001"),
        # Gst.Caps.from_string("video/x-raw,format=ARGB,width=(int)800,height=(int)320,framerate=(fraction)0/1"),
    )
    _src_template = Gst.PadTemplate.new(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string("video/x-raw,format=BGRx,width=(int)800,height=(int)320,framerate=(fraction)30000/1001"),
        # Gst.Caps.from_string("video/x-raw,format=BGRx,width=(int)800,height=(int)320,framerate=(fraction)0/1"),
    )
    __gsttemplates__ = (_src_template, _sink_template)  # Order can matter for some tools
    __gproperties__ = {
        # "batch-size": (int, "batch size", "batch size, currently supports only 1", 1, 8, 1, GObject.ParamFlags.READWRITE),
        "type": (str, "segment or detect", "Segmentation or detection", "segment", GObject.ParamFlags.READWRITE)
    }

    def __init__(self):
        super().__init__()
        # Initialize properties from defaults if not set otherwise
        self.batch_size = 1
        self.model = None
        self.model_task = "segment"
        self.device = None
        # self.initialize_device()

    def initialize_device(self):
        device_id = 0
        # print("########################################3 #######################3", self.model_task)
        self.device = ttnn.CreateDevice(
            device_id=0,
            dispatch_core_config=self.get_dispatch_core_config(),
            l1_small_size=79104,
            trace_region_size=23887872,
            num_command_queues=2,
        )
        self.device.enable_program_cache()
        # The updated runner in main requires a model_location_generator; pass None for local.
        self.model = UFLDPerformantRunner(self.device, model_location_generator=None)
        print("########################################", self.batch_size)

    def save_yolo_predictions_by_model(self, result, image, model_name):
        if model_name == "torch_model":
            bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
        else:
            bounding_box_color, label_color = (255, 0, 0), (255, 255, 0)

        boxes = result["boxes"]["xyxy"]
        scores = result["boxes"]["conf"]
        classes = result["boxes"]["cls"]
        names = result["names"]

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {score.item():.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        return image

    def get_dispatch_core_config(self):
        # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
        dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
        if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
            dispatch_core_type = ttnn.device.DispatchCoreType.ETH
        dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
        dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)

        return dispatch_core_config

    def _trace_release(self):
        ttnn.release_trace(self.device, self.tid)

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "type":
            return self.model_task  # Return the Python instance attribute
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == "type":
            self.model_task = value
            print("SET PROP", prop.name, value)
        else:
            raise AttributeError(f"Unknown property {prop.name}")

        self.initialize_device()

    def do_transform(self, inbuf: Gst.Buffer, outbuf: Gst.Buffer) -> Gst.FlowReturn:
        if self.model is None or self.device is None:
            self.initialize_device()
        try:
            success, in_map_info = inbuf.map(Gst.MapFlags.READ)
            if not success:
                Gst.error("UFLDv2: Failed to map input buffer")
                return Gst.FlowReturn.ERROR
            # frame_data1 = np.frombuffer(in_map_info.data, dtype=np.uint8).reshape(640, 640, 4)
            # print(np.frombuffer(in_map_info.data, dtype=np.uint8).shape)
            frame_data_bgrx = np.frombuffer(in_map_info.data, dtype=np.uint8).reshape(320, 800, 4)
            original_frame_for_drawing = frame_data_bgrx[:, :, :3].copy()
            # frame_data = frame_data1[:, :, :3].copy()
            # frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame_data_rgb = cv2.cvtColor(original_frame_for_drawing, cv2.COLOR_BGR2RGB)
            frame_data_rgb = np.array(frame_data_rgb)

            if type(frame_data_rgb) == np.ndarray and len(frame_data_rgb.shape) == 3:
                torch_frame_data = torch.from_numpy(frame_data_rgb).float().div(255.0).unsqueeze(0)
            elif type(frame_data_rgb) == np.ndarray and len(frame_data_rgb.shape) == 4:
                torch_frame_data = torch.from_numpy(frame_data_rgb).float().div(255.0)

            ts = time.time()
            torch_frame_data = torch.permute(torch_frame_data, (0, 3, 1, 2))

            # Apply the same preprocessing as standalone version
            # Convert to PIL Image for consistent transforms
            import torchvision.transforms as transforms
            from PIL import Image

            # Convert tensor back to PIL Image for proper preprocessing
            img_tensor = torch_frame_data.squeeze(0).permute(1, 2, 0)  # HWC format
            img_array = (img_tensor.numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)

            # Apply the exact same transforms as standalone
            target_size = (320, 800)
            crop_ratio = 0.8
            img_transforms = transforms.Compose(
                [
                    transforms.Resize((int(target_size[0] / crop_ratio), target_size[1])),  # (400, 800)
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

            img_tensor = img_transforms(pil_img)

            # Crop to target size (same as standalone)
            crop_size = target_size[0]
            img_tensor = img_tensor[:, -crop_size:, :]  # Crop from bottom

            # Add batch dimension
            torch_frame_data = img_tensor.unsqueeze(0)

            # print("INPUT",torch_frame_data.shape)

            # Minimal inline processing to overlay lanes using demo utilities
            # Execute model and build lane structures similar to demo path
            out = self.model.run(torch_frame_data)
            out = (
                ttnn.to_torch(out, mesh_composer=self.model.runner_infra.output_mesh_composer)
                .squeeze(dim=1)
                .squeeze(dim=1)
            )
            num_grid_row = 100
            num_cls_row = 56
            num_grid_col = 100
            num_cls_col = 41
            num_lane_on_row = 4
            num_lane_on_col = 4
            dim1 = num_grid_row * num_cls_row * num_lane_on_row
            dim2 = num_grid_col * num_cls_col * num_lane_on_col
            dim3 = 2 * num_cls_row * num_lane_on_row
            dim4 = 2 * num_cls_col * num_lane_on_col
            pred = {
                "loc_row": out[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lane_on_row),
                "loc_col": out[:, dim1 : dim1 + dim2].view(-1, num_grid_col, num_cls_col, num_lane_on_col),
                "exist_row": out[:, dim1 + dim2 : dim1 + dim2 + dim3].view(-1, 2, num_cls_row, num_lane_on_row),
                "exist_col": out[:, -dim4:].view(-1, 2, num_cls_col, num_lane_on_col),
            }
            import numpy as _np

            row_anchor = _np.linspace(160, 710, num_cls_row) / 720
            col_anchor = _np.linspace(0, 1, num_cls_col)
            lanes = generate_tusimple_lines(
                pred["loc_row"][0],
                pred["exist_row"][0],
                pred["loc_col"][0],
                pred["exist_col"][0],
                row_anchor=row_anchor,
                col_anchor=col_anchor,
                mode="4row",
            )
            # Overlay lanes
            outImage = original_frame_for_drawing.copy()
            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
            thickness = 3
            import cv2 as _cv2

            h_samples = _np.linspace(160, 710, 56)
            for lane_idx, lane_xs in enumerate(lanes):
                lane_xs = _np.array(lane_xs)
                valid_mask = lane_xs != -2
                valid_xs = lane_xs[valid_mask]
                valid_ys = h_samples[valid_mask]
                points = _np.array([valid_xs, valid_ys]).T.astype(_np.int32)
                if len(points) >= 2:
                    _cv2.polylines(
                        outImage, [points], isClosed=False, color=colors[lane_idx % len(colors)], thickness=thickness
                    )
            outImage = cv2.cvtColor(outImage, cv2.COLOR_BGR2BGRA)

            inbuf.unmap(in_map_info)

            success, out_map_info = outbuf.map(Gst.MapFlags.WRITE)
            if not success:
                Gst.error("UFLDv2: Failed to map output buffer for writing")
                return Gst.FlowReturn.ERROR

            # Ensure the output buffer has enough space for the transformed image
            required_size = outImage.nbytes
            if outbuf.get_size() < required_size:
                Gst.error(f"UFLDv2: Output buffer too small. Required: {required_size}, Actual: {outbuf.get_size()}")
                return Gst.FlowReturn.ERROR

            # Corrected line: assign bytes to the memoryview slice
            # This copies the byte data from outImage into the GStreamer buffer's memory
            out_map_info.data[:] = outImage.tobytes()

            # Update the output buffer's size to reflect the actual data written
            outbuf.set_size(required_size)

            # Important: Copy timestamps and other buffer properties from input to output
            outbuf.pts = inbuf.pts
            outbuf.dts = inbuf.dts
            outbuf.duration = inbuf.duration
            outbuf.offset = inbuf.offset
            outbuf.offset_end = inbuf.offset_end

            return Gst.FlowReturn.OK

        except Exception as e:
            Gst.error(f"UFLDv2: Error in transform: {e}")
            return Gst.FlowReturn.ERROR
        finally:
            # Ensure buffers are unmapped even if an error occurs
            if in_map_info is not None:
                inbuf.unmap(in_map_info)
            # if out_map_info is not None:
            #    outbuf.unmap(out_map_info)

    def do_start(self):
        try:
            if self.model is None or self.device is None:
                self.initialize_device()
            return True
        except Exception as e:
            Gst.error(f"UFLDv2: Failed to start: {e}")
            return False


GObject.type_register(UFLDv2)

__gstelementfactory__ = ("ufldv2", Gst.Rank.NONE, UFLDv2)

# The following are not strictly needed for __gstelementfactory__ but good for plugin tools
GST_PLUGIN_NAME = "ufldv2plugin"
__gstplugininit__ = None  # Not using a plugin_init function with __gstelementfactory__
