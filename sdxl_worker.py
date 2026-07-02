# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import time
import queue
import multiprocessing as mp
from sdxl_runner import SDXLRunner
from sdxl_config import SDXLConfig
from utils.logger import setup_logger


def setup_worker_environment(worker_id: int, config: SDXLConfig):
    """
    Set worker-specific environment variables

    Args:
        worker_id: Worker process ID
        config: SDXL configuration
    """
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.getcwd()

    # Set device visibility EARLY (before any ttnn operations)
    # This ensures the correct devices are visible when ttnn initializes
    device_ids_str = ",".join(map(str, config.device_ids))
    os.environ["TT_VISIBLE_DEVICES"] = device_ids_str
    os.environ["TT_METAL_VISIBLE_DEVICES"] = device_ids_str

    # Performance settings
    if config.is_galaxy:
        os.environ["TT_MM_THROTTLE_PERF"] = "5"

        # Set mesh graph descriptor for 1x1 Galaxy mode (matches tt-media-server)
        tt_metal_home = os.environ.get("TT_METAL_HOME", os.getcwd())
        if config.device_mesh_shape == (1, 1):
            os.environ[
                "TT_MESH_GRAPH_DESC_PATH"
            ] = f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto"
        elif config.device_mesh_shape == (2, 1):
            os.environ[
                "TT_MESH_GRAPH_DESC_PATH"
            ] = f"{tt_metal_home}/tt_metal/fabric/mesh_graph_descriptors/n300_mesh_graph_descriptor.textproto"


def device_worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    warmup_signal_queue: mp.Queue,
    kernel_ready_queue: mp.Queue,
    error_queue: mp.Queue,
    config: SDXLConfig,
):
    """
    Main worker process function that handles SDXL inference tasks

    Args:
        worker_id: Worker process ID
        task_queue: Queue for incoming tasks (task_id, request dict)
        result_queue: Queue for results (task_id, images, inference_time)
        warmup_signal_queue: Queue to signal warmup completion
        kernel_ready_queue: Queue to signal kernel compilation complete (for overlapped startup)
        error_queue: Queue for error reporting
        config: SDXL configuration

    Based on: /home/tt-admin/tt-inference-server/tt-media-server/model_services/device_worker.py
    """
    logger = setup_logger(f"Worker-{worker_id}")
    logger.info(f"Worker {worker_id} starting...")

    try:
        # Setup environment
        setup_worker_environment(worker_id, config)

        # Initialize runner
        runner = SDXLRunner(worker_id, config)
        runner.initialize_device()
        runner.load_model(kernel_ready_queue=kernel_ready_queue)

        # Signal warmup complete
        warmup_signal_queue.put(worker_id)
        logger.info(f"Worker {worker_id} ready for inference")

        # Main processing loop
        while True:
            try:
                # Get task from queue (blocking with timeout)
                task = task_queue.get(timeout=1)

                if task is None:  # Shutdown signal
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break

                task_id, request = task
                logger.info(f"Worker {worker_id} processing task {task_id}")

                # Run inference
                start_time = time.time()
                images = runner.run_inference([request])
                inference_time = time.time() - start_time

                # Put result in queue
                result_queue.put({"task_id": task_id, "images": images, "inference_time": inference_time})

                logger.info(f"Task {task_id} completed in {inference_time:.2f}s")

            except queue.Empty:
                # Normal timeout - no task available, just try again
                continue
            except Exception as e:
                import traceback

                logger.error(f"Error processing task: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                error_queue.put({"worker_id": worker_id, "error": str(e)})

        # Cleanup
        runner.close_device()
        logger.info(f"Worker {worker_id} shutdown complete")

    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")
        error_queue.put({"worker_id": worker_id, "error": f"Fatal: {str(e)}"})
