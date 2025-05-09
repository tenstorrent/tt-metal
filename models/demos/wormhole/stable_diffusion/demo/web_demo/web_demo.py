# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import signal
import subprocess
import sys

from loguru import logger

# Create the argument parser
parser = argparse.ArgumentParser(description="Stable Diffusion web demo")

# Add the 'port' argument with a default value of 7000
parser.add_argument("--port", type=int, default=7000, help="Port number to run the web demo on (default: 7000)")

# Parse the command-line arguments
args = parser.parse_args()

# Two scripts to run
script1 = f"pytest models/demos/wormhole/stable_diffusion/demo/web_demo/flaskserver.py --port {args.port} "
script2 = f"streamlit run models/demos/wormhole/stable_diffusion/demo/web_demo/streamlit_app.py -- --port {args.port}"

# Start both scripts using subprocess
process1 = subprocess.Popen(script1, shell=True)
process2 = subprocess.Popen(script2, shell=True)


# Function to kill process using port 5000
def kill_port_5000():
    try:
        result = subprocess.check_output("lsof -i :5000 | grep LISTEN | awk '{print $2}'", shell=True)
        pid = int(result.strip())
        logger.info(f"Killing process {pid} using port 5000")
        os.kill(pid, signal.SIGTERM)
    except subprocess.CalledProcessError:
        logger.error("No process found using port 5000")
    except Exception as e:
        logger.error(f"Error occurred: {e}")


# Function to terminate both processes and kill port 5000
def signal_handler(sig, frame):
    logger.info("Terminating processes...")
    process1.terminate()
    process2.terminate()
    kill_port_5000()
    logger.info("Processes terminated and port 5000 cleared.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

logger.info("Running. Press Ctrl+C to stop.")
try:
    process1.wait()
    process2.wait()
except KeyboardInterrupt:
    signal_handler(None, None)
