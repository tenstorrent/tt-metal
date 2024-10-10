import subprocess
import signal
import sys
import os

# Two scripts to run
script1 = "pytest models/demos/wormhole/stable_diffusion/demo/web_demo/sdserver.py"
script2 = "python models/demos/wormhole/stable_diffusion/demo/web_demo/flaskserver.py"
script3 = "streamlit run models/demos/wormhole/stable_diffusion/demo/web_demo/streamlit_app.py"

# Start both scripts using subprocess
process1 = subprocess.Popen(script1, shell=True)
process2 = subprocess.Popen(script2, shell=True)
process3 = subprocess.Popen(script3, shell=True)


# Function to kill process using port 5000
def kill_port_5000():
    try:
        result = subprocess.check_output("lsof -i :5000 | grep LISTEN | awk '{print $2}'", shell=True)
        pid = int(result.strip())
        print(f"Killing process {pid} using port 5000")
        os.kill(pid, signal.SIGTERM)
    except subprocess.CalledProcessError:
        print("No process found using port 5000")
    except Exception as e:
        print(f"Error occurred: {e}")


# Function to terminate both processes and kill port 5000
def signal_handler(sig, frame):
    print("Terminating processes...")
    process1.terminate()
    process2.terminate()
    kill_port_5000()
    print("Processes terminated and port 5000 cleared.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print("Running. Press Ctrl+C to stop.")
try:
    process1.wait()
    process2.wait()
except KeyboardInterrupt:
    signal_handler(None, None)
