# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from http import HTTPStatus
from threading import Thread

from flask import Flask, jsonify, request, send_from_directory
from gunicorn.app.base import BaseApplication
from loguru import logger

from models.demos.wormhole.stable_diffusion.demo.web_demo.model import warmup_model
from models.demos.wormhole.stable_diffusion.demo.web_demo.task_queue import TaskQueue

app = Flask(__name__)

# Initialize the task queue
task_queue = TaskQueue()


# worker thread to process the task queue
def create_worker():
    try:
        while True:
            # get task if one exists, otherwise block
            task_id = task_queue.get_task()
            if task_id:
                task_queue.process_task(task_id)
    except Exception as error:
        logger.error(error)


@app.route("/")
def hello_world():
    return jsonify({"message": "OK\n"}), 200


@app.route("/enqueue", methods=["POST"])
def enqueue_prompt():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), HTTPStatus.BAD_REQUEST

    # Enqueue the prompt and start processing
    task_id = task_queue.enqueue_task(prompt)
    return jsonify({"task_id": task_id, "status": "Enqueued"}), HTTPStatus.CREATED


@app.route("/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task_status = task_queue.get_task_status(task_id)
    if not task_status:
        return jsonify({"error": "Task not found"}), HTTPStatus.NOT_FOUND
    return jsonify({"task_id": task_id, "status": task_status["status"]}), HTTPStatus.OK


@app.route("/fetch_image/<task_id>", methods=["GET"])
def fetch_image(task_id):
    task_status = task_queue.get_task_status(task_id)
    if not task_status:
        return jsonify({"error": "Task not found"}), HTTPStatus.NOT_FOUND

    if task_status["status"] != "Completed":
        return jsonify({"error": "Task not completed yet"}), HTTPStatus.BAD_REQUEST

    image_path = task_status["image_path"]
    directory = os.getcwd()  # get the current working directory
    return send_from_directory(directory, image_path)


class GunicornApp(BaseApplication):
    def __init__(self, app, port):
        self.app = app
        self.port = port
        super().__init__()

    def load(self):
        return self.app

    def load_config(self):
        config = {
            "bind": f"0.0.0.0:{self.port}",  # Specify the binding address
            "workers": 1,  # Number of Gunicorn workers
            "reload": False,
            "worker_class": "gthread",
            "threads": 16,
            "post_worker_init": self.post_worker_init,
            "timeout": 0,
        }

        # Set the configurations for Gunicorn (optional but useful)
        for key, value in config.items():
            self.cfg.set(key, value)

    def post_worker_init(self, worker):
        # all setup tasks and spinup background threads must be performed
        # here as gunicorn spawns worker processes who must be the parent
        # of all server threads
        warmup_model()

        # run the model worker in a background thread
        thread = Thread(target=create_worker)
        thread.daemon = True
        thread.start()


def test_app(port):
    # Ensure the generated images directory exists
    os.makedirs("generated_images", exist_ok=True)
    gunicorn_app = GunicornApp(app, port)
    gunicorn_app.run()
