import sys
import os

os.environ["TERM"] = "xterm-256color"

import curses
import threading
import subprocess
import shlex
import re
import time
import signal
import psutil


def main(stdscr):
    curses.curs_set(0)  # Hide cursor
    curses.start_color()
    curses.use_default_colors()

    # Define color pairs using extended colors
    define_color_pairs()

    max_y, max_x = stdscr.getmaxyx()

    # Input fields positions (reordered)
    input_fields = [
        {"label": "Command [demo]", "value": "", "x": 0, "y": 0},
        {"label": "Model (1b, 3b, 8b) [all]", "value": "", "x": 0, "y": 1},
        {"label": "Device (n150, n300, t3k) [all]", "value": "", "x": 0, "y": 2},
    ]

    output_entries = []
    current_line = 0  # Index of the current line (input fields + output entries)
    total_lines = len(input_fields)

    screen_lock = threading.Lock()
    screen_needs_update = threading.Event()  # New event to signal screen updates
    last_drawn_state = {
        "input_fields": [],  # Start with an empty list
        "output_entries": [],
        "current_line": -1,  # Set to an invalid value to force initial draw
        "max_y": max_y,
        "max_x": max_x,
    }

    # Start the worker thread
    worker_stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=worker_thread_func, args=(output_entries, worker_stop_event, screen_lock, screen_needs_update)
    )
    worker_thread.daemon = True
    worker_thread.start()

    stdscr.nodelay(True)  # Set getch() non-blocking

    # Initial draw
    draw_changes(stdscr, input_fields, output_entries, current_line, last_drawn_state)
    stdscr.refresh()

    exiting = False  # New flag to indicate we're in the process of exiting

    # Main loop
    while True:
        new_max_y, new_max_x = stdscr.getmaxyx()
        if new_max_y != last_drawn_state["max_y"] or new_max_x != last_drawn_state["max_x"]:
            stdscr.clear()
            last_drawn_state["max_y"], last_drawn_state["max_x"] = new_max_y, new_max_x
            last_drawn_state["input_fields"] = []  # Reset to force redraw
            last_drawn_state["output_entries"] = []  # Reset to force redraw
            screen_needs_update.set()

        if screen_needs_update.is_set():
            with screen_lock:
                # Draw everything
                draw_changes(stdscr, input_fields, output_entries, current_line, last_drawn_state)
                stdscr.refresh()
            screen_needs_update.clear()

        c = stdscr.getch()

        # Check if we should exit after all jobs are done
        if exiting and all(
            entry["status"] in ["Exiting", "Cancelled", "Error", "Finished"] for entry in output_entries
        ):
            return

        if c == -1:
            # No key pressed, continue to next iteration
            time.sleep(0.01)  # Short sleep to prevent high CPU usage
            continue
        elif c == 27:  # Handle escape key press
            if not exiting:
                exiting = True
                worker_stop_event.set()

                # Find the running job and set it to terminate
                running_entry = None
                for entry in output_entries:
                    with entry["lock"]:
                        if entry["process"] and entry["process"].poll() is None:
                            running_entry = entry
                            entry["stop_event"].set()
                            terminate_process_tree(entry["process"].pid)
                            entry["status"] = "Terminating"
                            break

                # Set all other jobs to "Exiting"
                for entry in output_entries:
                    with entry["lock"]:
                        if entry != running_entry and entry["status"] == "Waiting":
                            entry["status"] = "Exiting"

                # Clear input fields
                for field in input_fields:
                    field["value"] = "Exiting"

                screen_needs_update.set()
            else:
                # If escape is pressed again while exiting, force quit
                return
        elif c == curses.KEY_UP:
            current_line = (current_line - 1) % (len(input_fields) + len(output_entries))
            screen_needs_update.set()
        elif c == curses.KEY_DOWN:
            current_line = (current_line + 1) % (len(input_fields) + len(output_entries))
            screen_needs_update.set()
        elif c == curses.KEY_ENTER or c == 10 or c == 13:
            if not exiting:
                if current_line < len(input_fields):
                    # We are in input fields
                    current_field = current_line

                    # If the last field is selected, submit the command
                    if current_field == len(input_fields) - 1:
                        # Submit command
                        command_input = input_fields[0]["value"] or "demo"
                        model_input = input_fields[1]["value"] or "1b,3b,8b"
                        device_input = input_fields[2]["value"] or "n150,n300,t3k"

                        # Parse models and devices
                        models = parse_list(model_input)
                        devices = parse_list(device_input)

                        # Generate combinations (reordered)
                        combinations = [(m, d) for m in models for d in devices]

                        # Create output entries
                        for model, device in combinations:
                            command_name = get_command_name(command_input)
                            entry = {
                                "command_name": command_name,
                                "model": model,
                                "device": device.upper(),
                                "status": "Waiting",
                                "output": "",
                                "process": None,
                                "log_file": None,
                                "index": len(output_entries),
                                "stop_event": threading.Event(),
                                "lock": threading.Lock(),
                                "command_input": command_input,  # Save the command input
                            }
                            output_entries.append(entry)
                        # Update total_lines
                        total_lines = len(input_fields) + len(output_entries)
                        current_line = 0
                        screen_needs_update.set()
                    else:
                        # Otherwise if not the last field, move to next field
                        current_line = (current_line + 1) % total_lines
                        screen_needs_update.set()
                else:
                    # We are in the output entries
                    entry_index = current_line - len(input_fields)
                    if entry_index < len(output_entries):
                        entry = output_entries[entry_index]
                        if entry["log_file"]:
                            # Save current terminal state
                            curses.def_prog_mode()
                            # Exit curses temporarily
                            curses.endwin()
                            # Run less command
                            os.system(f"less -R {entry['log_file'].name}")
                            # Resume curses
                            curses.reset_prog_mode()
                            stdscr.refresh()
                            screen_needs_update.set()
            else:
                # Ignore enter key when exiting
                continue
        elif c == curses.KEY_BACKSPACE or c == 127 or c == ord("x"):
            if current_line < len(input_fields):
                current_field = current_line
                # Remove last character from current field
                if len(input_fields[current_field]["value"]) > 0:
                    input_fields[current_field]["value"] = input_fields[current_field]["value"][:-1]
            else:
                # We are in the output entries
                entry_index = current_line - len(input_fields)
                if entry_index < len(output_entries):
                    entry = output_entries[entry_index]
                    with entry["lock"]:
                        if entry["process"] and entry["process"].poll() is None:
                            # Cancel the running process
                            entry["stop_event"].set()
                            terminate_process_tree(entry["process"].pid)
                            entry["status"] = "Terminating"
                        elif entry["status"] in ["Cancelled", "Error", "Finished"]:
                            # Remove the entry if it's already cancelled
                            output_entries.pop(entry_index)
                            total_lines -= 1
                            if current_line >= total_lines:
                                current_line = total_lines - 1
                        else:
                            # Set to cancelled if not running and not already cancelled
                            entry["status"] = "Cancelled"
                            current_line = (current_line + 1) % total_lines
                    screen_needs_update.set()
        elif c == 9:  # Tab key
            current_line = (current_line + 1) % total_lines
            screen_needs_update.set()
        else:
            if current_line < len(input_fields) and not exiting:
                current_field = current_line
                input_fields[current_field]["value"] += chr(c)
            screen_needs_update.set()


def define_color_pairs():
    # Extended color codes (assuming 256-color support)
    # Muted pastel colors
    COLOR_LIGHT_BLUE = 109  # Light pastel blue
    COLOR_LIGHT_CYAN = 152  # Light pastel cyan
    COLOR_LIGHT_GREEN = 108  # Light pastel green
    COLOR_LIGHT_YELLOW = 229  # Light pastel yellow
    COLOR_LIGHT_RED = 174  # Light pastel red
    COLOR_LIGHT_PURPLE = 183  # Light pastel purple
    COLOR_GRAY = 250  # Light gray
    COLOR_WHITE = 15  # Bright white
    COLOR_BLACK = 16  # Black

    # Initialize color pairs
    curses.init_pair(1, COLOR_BLACK, COLOR_GRAY)  # Selected field/background
    curses.init_pair(2, COLOR_LIGHT_CYAN, -1)  # Labels
    curses.init_pair(3, COLOR_WHITE, -1)  # Input values
    curses.init_pair(4, COLOR_LIGHT_GREEN, -1)  # Header text
    curses.init_pair(5, COLOR_GRAY, -1)  # 'Waiting' status
    curses.init_pair(6, COLOR_LIGHT_YELLOW, -1)  # 'Running' status
    curses.init_pair(7, COLOR_LIGHT_GREEN, -1)  # 'Finished' status
    curses.init_pair(8, COLOR_LIGHT_RED, -1)  # 'Error' status

    # Store the color pair numbers for use in the rest of the program
    global COLOR_PAIR_SELECTED
    global COLOR_PAIR_LABEL
    global COLOR_PAIR_VALUE
    global COLOR_PAIR_HEADER
    global COLOR_PAIR_WAITING
    global COLOR_PAIR_RUNNING
    global COLOR_PAIR_FINISHED
    global COLOR_PAIR_ERROR
    global COLOR_PAIR_SPEED

    COLOR_PAIR_SELECTED = curses.color_pair(1)
    COLOR_PAIR_LABEL = curses.color_pair(2)
    COLOR_PAIR_VALUE = curses.color_pair(3)
    COLOR_PAIR_HEADER = curses.color_pair(4) | curses.A_BOLD
    COLOR_PAIR_WAITING = curses.color_pair(5)
    COLOR_PAIR_RUNNING = curses.color_pair(6)
    COLOR_PAIR_FINISHED = curses.color_pair(7)
    COLOR_PAIR_ERROR = curses.color_pair(8)
    COLOR_PAIR_SPEED = curses.color_pair(6)  # Use the same color as RUNNING


def draw_changes(stdscr, input_fields, output_entries, current_line, last_drawn_state):
    max_y, max_x = stdscr.getmaxyx()

    # Update input fields
    for idx, field in enumerate(input_fields):
        if (
            idx >= len(last_drawn_state["input_fields"])
            or field != last_drawn_state["input_fields"][idx]
            or current_line != last_drawn_state["current_line"]
        ):
            draw_input_field(stdscr, field, idx == current_line, max_x)
            if idx < len(last_drawn_state["input_fields"]):
                last_drawn_state["input_fields"][idx] = field.copy()
            else:
                last_drawn_state["input_fields"].append(field.copy())

    # Draw a divider line
    divider_y = len(input_fields)
    stdscr.hline(divider_y, 0, curses.ACS_HLINE, max_x)

    # Draw header
    header_y = divider_y + 1
    header = format_header(max_x)
    stdscr.addstr(header_y, 0, header, COLOR_PAIR_HEADER)
    stdscr.clrtoeol()

    # Update output entries
    output_start_y = header_y + 1
    for idx, entry in enumerate(output_entries):
        y = output_start_y + idx
        if y >= max_y - 2:
            break
        if (
            idx >= len(last_drawn_state["output_entries"])
            or entry != last_drawn_state["output_entries"][idx]
            or current_line != last_drawn_state["current_line"]
        ):
            draw_output_entry(stdscr, entry, y, current_line == len(input_fields) + idx, max_x)

    # Clear any extra lines if output entries were removed
    for y in range(
        output_start_y + len(output_entries), min(output_start_y + len(last_drawn_state["output_entries"]), max_y - 2)
    ):
        stdscr.move(y, 0)
        stdscr.clrtoeol()

    # Update last_drawn_state
    last_drawn_state["output_entries"] = [entry.copy() for entry in output_entries]
    last_drawn_state["current_line"] = current_line


def draw_input_field(stdscr, field, is_selected, max_x):
    x, y = field["x"], field["y"]
    label, value = field["label"], field["value"]
    if is_selected:
        stdscr.addstr(y, x, label + ": ", COLOR_PAIR_SELECTED)
        stdscr.addstr(y, x + len(label) + 2, value, COLOR_PAIR_SELECTED)
    else:
        stdscr.addstr(y, x, label + ": ", COLOR_PAIR_LABEL)
        stdscr.addstr(y, x + len(label) + 2, value, COLOR_PAIR_VALUE)
    stdscr.clrtoeol()  # Clear the rest of the line


def draw_output_entry(stdscr, entry, y, is_selected, max_x):
    cols = [
        entry["command_name"],
        entry["model"],
        entry["device"],
        entry["status"],
        entry.get("speed", ""),
        entry["output"],
    ]
    col_widths = [15, 10, 10, 20, 10, max_x - 70]  # Adjusted widths to accommodate the speed column

    x = 0
    for i, (col, width) in enumerate(zip(cols, col_widths)):
        col_text = str(col)[:width].ljust(width)
        if is_selected:
            stdscr.addstr(y, x, col_text, COLOR_PAIR_SELECTED)
        else:
            color = curses.color_pair(0)
            if i == 3:  # Status column
                status = entry["status"]
                if status == "Waiting" or status == "Cancelled":
                    color = COLOR_PAIR_WAITING
                elif status in ["Running", "Initializing device", "Prefill", "Decode", "Starting"] or status.startswith(
                    "Loading "
                ):
                    color = COLOR_PAIR_RUNNING
                elif status == "Finished":
                    color = COLOR_PAIR_FINISHED
                elif status == "Error":
                    color = COLOR_PAIR_ERROR
                elif status == "Terminating" or status == "Resetting":
                    color = COLOR_PAIR_WAITING
            elif i == 4:  # Speed column
                color = COLOR_PAIR_SPEED
            else:
                color = curses.color_pair(0)
            stdscr.addstr(y, x, col_text, color)
        x += width
    stdscr.clrtoeol()  # Clear the rest of the line


def format_header(max_x):
    cols = ["Command", "Model", "Device", "Status", "Speed", "Output"]
    col_widths = [15, 10, 10, 20, 10, max_x - 70]  # Adjusted widths to accommodate the speed column
    formatted_cols = []
    for col, width in zip(cols, col_widths):
        formatted_cols.append(col[:width].ljust(width))
    return "".join(formatted_cols)


def parse_list(input_str):
    if not input_str.strip():
        return [""]
    else:
        items = [item.strip() for item in re.split(r"[,\s]+", input_str.strip()) if item.strip()]
        return items


def worker_thread_func(output_entries, stop_event, screen_lock, screen_needs_update):
    while not stop_event.is_set():
        running_entry = None
        for entry in output_entries:
            with entry["lock"]:
                if entry["process"] and entry["process"].poll() is None:
                    running_entry = entry
                    break

        if not running_entry:
            for entry in output_entries:
                with entry["lock"]:
                    if entry["status"] == "Waiting":
                        run_entry_command(entry, screen_lock, output_entries, screen_needs_update)
                        break
        # Set screen_needs_update whenever there's a change in output entries
        screen_needs_update.set()
        time.sleep(0.1)


def run_entry_command(entry, screen_lock, output_entries, screen_needs_update):
    entry["status"] = "Initializing device"
    screen_needs_update.set()

    # Set environment variables
    env = os.environ.copy()
    env["FAKE_DEVICE"] = entry["device"]
    env["LLAMA_DIR"] = get_llama_dir(entry["model"])

    # Prepare the command
    cmd_list = shlex.split(entry["command_input"])

    # Open log file
    log_filename = get_log_filename(entry["device"], entry["model"], entry["command_input"])
    os.makedirs("logs", exist_ok=True)
    entry["log_file"] = open(os.path.join("logs", log_filename), "w")

    # Start the subprocess
    entry["process"] = subprocess.Popen(
        cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, preexec_fn=os.setsid
    )

    # Read the output in a separate thread
    entry["thread"] = threading.Thread(
        target=process_output, args=(entry, screen_lock, output_entries, screen_needs_update)
    )
    entry["thread"].daemon = True
    entry["thread"].start()


def process_output(entry, screen_lock, output_entries, screen_needs_update):
    process = entry["process"]
    log_file = entry["log_file"]
    previous_line = ""
    try:
        for line in iter(process.stdout.readline, ""):
            if entry["stop_event"].is_set():
                break
            # Write to log file
            log_file.write(line)
            log_file.flush()

            # Update status and output based on output
            status, output, speed = parse_output_line(line, previous_line, entry["status"])
            # Append input and output of parse_output_line to a log file
            with open("parse_output_log.txt", "a") as parse_log:
                parse_log.write(f"Input: {line.strip()}, Previous status: {entry['status']}\n")
                parse_log.write(f"Output: Status: {status}, Output: {output}, Speed: {speed}\n\n")
            previous_line = line.strip()
            with entry["lock"]:
                if status != entry["status"] or output or speed is not None:
                    entry["status"] = status
                    if output:
                        entry["output"] = output
                    if speed is not None:
                        entry["speed"] = f"{speed:.1f}"
                    screen_needs_update.set()

            with screen_lock:
                pass  # Screen will be updated in main loop

    finally:
        # Ensure we close the stdout stream
        process.stdout.close()

        # Wait for the process to fully terminate
        process.wait()

        with entry["lock"]:
            if entry["status"] == "Terminating":
                entry["status"] = "Resetting"
                screen_needs_update.set()
                reset_device(entry)
                entry["status"] = "Cancelled"
            elif process.returncode != 0:
                entry["status"] = "Resetting"
                screen_needs_update.set()
                reset_device(entry)
                entry["status"] = "Error"
                # Try to find exception name in the log
                exception_name = find_exception_in_log(entry["log_file"].name)
                if exception_name:
                    entry["output"] = exception_name
            else:
                entry["status"] = "Finished"
        entry["process"] = None
        log_file.close()

        screen_needs_update.set()  # Ensure screen is updated after process termination

    # Start the next waiting entry
    for next_entry in output_entries:
        if next_entry["index"] > entry["index"]:
            with next_entry["lock"]:
                if next_entry["status"] == "Waiting" and not any(
                    e["process"] and e["process"].poll() is None for e in output_entries
                ):
                    run_entry_command(next_entry, screen_lock, output_entries, screen_needs_update)
                    break


def parse_output_line(line, previous_line, current_status):
    line = line.strip()
    if "Initializing device" in line:
        return "Initializing device", None, None
    elif "Loading weights" in line:
        return "Loading weights", None, None
    elif re.search(r"layers\.\d+\.", line):
        match = re.search(r"layers\.(\d+)\.", line)
        if match:
            layer_number = match.group(1)
            return f"Loading layer {layer_number}", None, None
    elif "Starting inference..." in line:
        return "Starting", None, None
    elif "Starting prefill..." in line:
        return "Prefill", None, None
    elif "Starting decode..." in line:
        return "Decode", None, None
    elif line == "output:":
        return "Waiting for output", None, None
    elif current_status == "Waiting for output" and previous_line == "output:":
        if "<|start_header_id|>assistant<|end_header_id|>" in line:
            output = line.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1].strip()
            if output:
                return "Running", output, None
            else:
                return "Assistant output", None, None  # wait for a non-blank line
        else:
            return "Running", line, None
    elif current_status == "Assistant output" and line:  # skip blank lines
        return "Running", line, None

    # Check for speed information
    speed_match = re.search(r"@ (\d+\.\d+) tok/s/user", line)
    if speed_match:
        speed = float(speed_match.group(1))
        return current_status, None, speed

    return current_status, None, None


def get_llama_dir(model):
    llama_dir = {
        "1b": os.environ.get("LLAMA_32_1B_DIR", "/proj_sw/user_dev/llama32-data/Llama3.2-1B-Instruct"),
        "3b": os.environ.get("LLAMA_32_3B_DIR", "/proj_sw/user_dev/llama32-data/Llama3.2-3B-Instruct"),
        "8b": os.environ.get("LLAMA_31_8B_DIR", "/proj_sw/user_dev/llama31-8b-data/Meta-Llama-3.1-8B-Instruct"),
    }.get(model, "")

    if not llama_dir or not os.path.exists(llama_dir):
        print(f"Error: The directory for the {model} model does not exist: {llama_dir}")
        print("You can set the following environment variables to specify the correct directory path:")
        print("  - LLAMA_32_1B_DIR for 1b model")
        print("  - LLAMA_32_3B_DIR for 3b model")
        print("  - LLAMA_31_8B_DIR for 8b model")
        sys.exit(1)

    return llama_dir


def get_command_name(command_input):
    # Get command name
    if "pytest" in command_input:
        match = re.search(r"pytest\s+([\S]+)", command_input)
        if match:
            test_file = match.group(1)
            basename = os.path.basename(test_file).split(".")[0]
            command_name = basename
        else:
            command_name = "pytest"
    else:
        cmd = shlex.split(command_input)[0]
        command_name = os.path.basename(cmd)
    return command_name


def get_log_filename(device, model, command_input):
    command_name = get_command_name(command_input)
    filename = f"{device}-{model}-{command_name}.log"
    filename = filename.replace("/", "_")
    return filename


def find_exception_in_log(log_filename):
    exception_name = None
    with open(log_filename, "r") as f:
        log_lines = f.readlines()
        for line in reversed(log_lines):
            # Check for Python exceptions
            match = re.search(r"(\w+Error):", line)
            if match:
                exception_name = match.group(1)
                break

            # Check for TT_FATAL errors
            tt_fatal_match = re.search(r"TT_FATAL\s*(.+)", line)
            if tt_fatal_match:
                exception_name = tt_fatal_match.group(1).strip()
                break

            # Check for other FATAL errors
            fatal_match = re.search(r"FATAL", line)
            if fatal_match:
                parts = line.split("|", 1)
                if len(parts) > 1:
                    exception_name = parts[1].strip()
                break
    return exception_name


def terminate_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass  # Process already terminated


def reset_device(entry):
    reset_cmd = ["tt-smi", "-wr", "all"]
    try:
        subprocess.run(reset_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        entry["output"] = f"Reset failed: {e.stderr.strip()}"


if __name__ == "__main__":
    curses.wrapper(main)
