# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import queue
import uuid

from loguru import logger

from models.demos.wormhole.stable_diffusion.demo.web_demo.model import generate_image_from_prompt


class TaskQueue:
    def __init__(self):
        self.tasks = {}
        self.task_queue = queue.Queue()

    def enqueue_task(self, prompt):
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "prompt": prompt,
            "status": "Pending",
            "image_path": None,
        }

        # Add the task to the queue for processing
        self.task_queue.put(task_id)
        return task_id

    def get_task(self):
        # Get the next task from the queue (this will block until there's a task)
        return self.task_queue.get()

    def process_task(self, task_id):
        task = self.tasks[task_id]
        task["status"] = "In Progress"

        # Simulate image generation (you can replace this with actual image generation code)
        logger.info(f"Processing task {task_id} for prompt: {task['prompt']}")
        image_path = generate_image_from_prompt(task["prompt"])

        # Update task status and store image path
        task["status"] = "Completed"
        task["image_path"] = image_path

        logger.info(f"Task {task_id} completed, image saved at {image_path}")

    def get_task_status(self, task_id):
        return self.tasks.get(task_id)
