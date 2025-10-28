#!/usr/bin/env python3
import sys
import gi
import signal
import os
import time  # Import time module for FPS calculation & processing simulation
import numpy as np
import cv2
import torch

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase
import ttnn

from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import is_wormhole_b0, torch2tt_tensor, is_blackhole
from models.utility_functions import (
    is_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_random,
    profiler,
)
from models.perf.perf_utils import prep_perf_report

if is_blackhole():
    from models.experimental.blackhole.ufld_v2_rn18like.tests.ufld_v2_rn18like_e2e_performant import UFLDv2Trace2CQ
else:
    from models.experimental.ufld_v2_rn18like.tests.ufld_v2_rn18like_e2e_performant import UFLDv2Trace2CQ

# Initialize GStreamer (only once)
Gst.init(None)


# --- Element Class Definition ---
class UFLD(GstBase.BaseTransform):
    # Element metadata (for GStreamer)

    __gtype_name__ = "GstUFLDPythonBatching"

    __gstmetadata__ = (
        "UFLD Python",  # Long name
        "Filter/Effect/Converter",  # Classification
        "Prepends a configurable string to text buffer data",  # Description
        "Your Name <your.email@example.com>",  # Author
    )

    # Pad Templates
    _sink_template = Gst.PadTemplate.new("sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
    _src_template = Gst.PadTemplate.new("src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any())
    __gsttemplates__ = (_src_template, _sink_template)  # Order can matter for some tools

    # Properties
    # To make properties work correctly with GObject.type_register and __gstelementfactory__,
    # they are best defined using GObject.Property
    # For simplicity in this direct adaptation, let's initialize them in __init__
    # and assume they are set programmatically or via default.
    # A more robust way would be to define them via GObject.Property if using this style.
    # However, __gproperties__ might still work if GObject.type_register is smart enough.
    # Let's try with __gproperties__ first as it's common.

    # __gproperties__ = {
    #    "batch-size": (
    #        GObject.TYPE_INT,     # Type of the property
    #        "Batch Size",         # Nickname (human-readable)
    #        "The processing batch size for the model (used at initialization).", # Blurb (description)
    #        1,                    # Default value
    #        GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT_ONLY # Flags
    #        # CONSTRUCT_ONLY means it can only be set when the element is created.
    #        # If you want to change it at runtime and reconfigure the model,
    #        # remove CONSTRUCT_ONLY and add complex logic in do_set_property.
    #        # For batch size of a model, CONSTRUCT_ONLY is often safer.
    #    )
    # }

    __gproperties__ = {
        "batch-size": (int, "Frequency", "Frequency of test signal", 1, 8, 1, GObject.ParamFlags.READWRITE),
        "stats-interval-ms": (
            int,
            "Stats interval (ms)",
            "Interval in milliseconds to report average preprocess/inference times",
            10,
            60000,
            500,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # Initialize properties from defaults if not set otherwise
        # self.batch_size = __gproperties__["batch-size"][3] # Default value
        self.model = None
        self.device = None
        self.height = 320
        self.width = 800
        self.batch_size = 1
        self.stats_interval_ms = 500
        # Stats accumulators
        self._stats_last_report_time = time.time()
        self._stats_frames = 0
        self._acc_pre_s = 0.0
        self._acc_inf_s = 0.0

        # try:
        #    device_id = 0
        #    self.device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=6397952, num_command_queues=2)

    #       #     self.batch_size=1
    #    ttnn.enable_program_cache(self.device)
    #    self.model = UFLDTrace2CQ()
    #    print("########################################", self.batch_size)
    #    self.model.initialize_ufld_trace_2cqs_inference(self.device,self.batch_size)

    # except Exception as e:
    #    Gst.exception_object(self, "UFLD: CRITICAL ERROR DURING __init__", exc=e)
    #    # It's important that GStreamer knows the element failed to initialize.
    #    # Re-raising the exception is one way to do this.
    #    # Depending on how GStreamer handles exceptions from __init__ in Python plugins,
    #    # the element might not be instantiated, or it might be in a broken state.
    #    raise # Re-raise the exception

    def initialize_device(self, batch_size=1):
        device_id = 0
        self.device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=6397952, num_command_queues=2)
        #        self.batch_size=1
        self.model = UFLDv2Trace2CQ()
        print("########################################", batch_size)
        self.model.initialize_ufldv2_trace_2cqs_inference(self.device, batch_size)

    def _trace_release(self):
        ttnn.release_trace(self.device, self.tid)

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "batch-size":
            return self.batch_size  # Return the Python instance attribute
        elif prop.name == "stats-interval-ms":
            return self.stats_interval_ms
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.GParamSpec, value=1):
        if prop.name == "batch-size":
            # Gst.info_object(self, f"Property 'batch-size' being set to: {value}")
            self.batch_size = value  # Store in Python instance attribute
            print("SET PROP", prop.name, value)
            # If batch_size change requires model re-initialization and it's not CONSTRUCT_ONLY:
            # if hasattr(self, 'model') and self.model is not None:
            #     Gst.warning_object(self, "Batch size changed. Re-initializing model.")
            #     try:
            #         self.model.initialize_ufld_trace_2cqs_inference(self.device, self.batch_size)
            #     except Exception as e:
            #         Gst.exception_object(self, "Failed to re-initialize model on batch-size change", exc=e)
            #         # Handle error appropriately, maybe mark element as broken
            self.initialize_device(self.batch_size)
        elif prop.name == "stats-interval-ms":
            self.stats_interval_ms = int(value)
            print("SET PROP", prop.name, value)
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_transform(self, inbuf: Gst.Buffer, outbuf: Gst.Buffer) -> Gst.FlowReturn:
        # print(self.batch_size)
        try:
            t0 = time.time()
            success, in_map_info = inbuf.map(Gst.MapFlags.READ)
            if not success:
                Gst.error("UFLD: Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            # original_data = in_map_info.data
            # prefix_bytes = self.prefix.encode('utf-8')
            # transformed_data = prefix_bytes + original_data
            frame_data = np.frombuffer(in_map_info.data, dtype=np.uint8).reshape(self.height, self.width, 3)
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            if type(frame_data) == np.ndarray and len(frame_data.shape) == 3:  # cv2 image
                frame_data = torch.from_numpy(frame_data.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(frame_data) == np.ndarray and len(frame_data.shape) == 4:
                frame_data = torch.from_numpy(frame_data.transpose(0, 3, 1, 2)).float().div(255.0)

            if self.batch_size > 1:
                n, c, h, w = frame_data.shape
                frame_data = frame_data.expand(self.batch_size, c, h, w)
            #            out, t = self.trace_run(frame_data)
            t1 = time.time()
            out = self.model.run_traced_inference(frame_data)
            t2 = time.time()

            inbuf.unmap(in_map_info)

            success, out_map_info = outbuf.map(Gst.MapFlags.WRITE)
            if not success:
                Gst.error("UFLD: Failed to map output buffer for writing")
                return Gst.FlowReturn.ERROR

            outbuf.unmap(out_map_info)

            # Accumulate stats and print at configured interval to match fpsdisplaysink cadence
            self._acc_pre_s += t1 - t0
            self._acc_inf_s += t2 - t1
            self._stats_frames += 1
            now = time.time()
            interval_s = self.stats_interval_ms / 1000.0
            if now - self._stats_last_report_time >= interval_s:
                frames = self._stats_frames if self._stats_frames > 0 else 1
                avg_pre = self._acc_pre_s / frames
                avg_inf = self._acc_inf_s / frames
                fps = frames / max(now - self._stats_last_report_time, 1e-6)
                print(
                    f"stats interval {self.stats_interval_ms}ms: frames={frames}, preprocess_avg={avg_pre:.6f}s, inference_avg={avg_inf:.6f}s, fps={fps:.1f}"
                )
                # Reset accumulators
                self._acc_pre_s = 0.0
                self._acc_inf_s = 0.0
                self._stats_frames = 0
                self._stats_last_report_time = now
            return Gst.FlowReturn.OK

        except Exception as e:
            Gst.error(f"UFLD: Error in transform: {e}")
            try:
                if "in_map_info" in locals():
                    inbuf.unmap(in_map_info)
            except Exception:
                pass
            try:
                if "out_map_info" in locals():
                    outbuf.unmap(out_map_info)
            except Exception:
                pass
            return Gst.FlowReturn.ERROR


# --- GObject Type Registration ---
# This explicitly registers the Python class with the GObject type system.
# After this call, UFLD.get_type() would be available.
GObject.type_register(UFLD)

# --- GStreamer Element Factory ---
# This tuple is what GStreamer's Python plugin loader looks for
# to make the element available.
# ("factory-name", rank, PythonClass)
# The factory-name should ideally be unique and all lowercase.
__gstelementfactory__ = ("ufld", Gst.Rank.NONE, UFLD)

# The following are not strictly needed for __gstelementfactory__ but good for plugin tools
GST_PLUGIN_NAME = "ufldplugin"
__gstplugininit__ = None  # Not using a plugin_init function with __gstelementfactory__
# Metadata for the plugin file itself (distinct from element metadata)
# This isn't formally used by __gstelementfactory__ in the same way as Gst.Plugin.register_static
# but can be good practice to include. For a pure __gstelementfactory__ plugin, GStreamer
# mainly cares about that tuple.
# However, gst-inspect-1.0 might pick up __gstmetadata__ if it's at the top level of the plugin file.
# For consistency with your working example, let's keep it minimal for now.
# If you want full plugin metadata to show with gst-inspect on the *plugin file*,
# the Gst.Plugin.register_static approach with plugin_init is more standard.
# But for just getting the element to work, __gstelementfactory__ is simpler.
