#!/bin/bash

install_dependencies() {
    # Update package list
    sudo apt update

    # Install dependencies
    sudo apt install -y \
        software-properties-common=0.99.9.12 \
        build-essential=12.8ubuntu1.1 \
        python3.8-venv=3.8.10-0ubuntu1~20.04.9 \
        libgoogle-glog-dev=0.4.0-1build1 \
        libyaml-cpp-dev=0.6.2-4ubuntu1 \
        libboost-all-dev=1.71.0.0ubuntu2 \
        libsndfile1=1.0.28-7ubuntu0.2 \
        libhwloc-dev

    # Check if installation was successful
    if [ $? -eq 0 ]; then
        echo "Dependencies installed successfully."
    else
        echo "Error: Failed to install dependencies."
        return 1
    fi
}

create_activate_venv() {
    # Check if .tools_env directory exists in the home directory
    if [ -d "$HOME/.tools_env" ]; then
        echo "Virtual environment already exists. Activating..."
        source "$HOME/.tools_env/bin/activate"
    else
        # Install python3-venv
        sudo apt install -y python3-venv || { echo "Failed to install python3-venv."; return 1; }

        # Create virtual environment in the home directory
        python3 -m venv "$HOME/.tools_env" || { echo "Failed to create virtual environment."; return 1; }

        # Activate virtual environment
        source "$HOME/.tools_env/bin/activate"

        echo "Virtual environment created and activated."
    fi
}

install_rustup() {
    # Download and install Rustup
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || { echo "Failed to install Rustup."; return 1; }

    # Add Rust to PATH
    source "$HOME/.cargo/env" || { echo "Failed to add Rust to PATH."; return 1; }
}

execute_function_with_timer() {
    # Start the timer
    start=$(date +%s)

    # Call the function passed as an argument and redirect its output and errors to a temporary file
    tmp_output=$(mktemp)
    "$@" >"$tmp_output" 2>&1 &

    # Get the PID of the background process
    pid=$!

    # Display the function name
    printf "Running %-20s" "$1"

    # Keep checking if the process is still running
    while kill -0 $pid 2>/dev/null; do
        # Print elapsed time with right alignment
        printf "\033[60G%02d:%02d:%02d" $(($SECONDS/3600)) $(($SECONDS%3600/60)) $(($SECONDS%60))
        sleep 1
    done

    wait $pid
    exit_status=$?

    # Read the output from the temporary file
    output=$(<"$tmp_output")

    # Remove the temporary file
    rm "$tmp_output"

    # Calculate the elapsed time
    end=$(date +%s)
    elapsed=$((end - start))

    # If the exit status is non-zero, display the error message
    if [ $exit_status -ne 0 ]; then
        printf "\033[60G%27s" ""
        printf "\033[60G\u2717"
        printf "\n"
        echo "Error: $output"
    else
        # Clear the timer value and print a checkmark
        printf "\033[60G%27s" ""
        printf "\033[60G\u2713"
        printf "\n"
    fi

    return $exit_status
}

# Install dependencies if not already installed
execute_function_with_timer install_dependencies

if [ $? -ne 0 ]; then
    echo "Error: install_dependencies failed."
    echo "Abort"
    exit 1
fi

# Activate virtual environment
create_activate_venv

if [ $? -ne 0 ]; then
    echo "Error: create_activate_venv failed."
    echo "Abort"
    exit 1
fi

# Install rust
execute_function_with_timer install_rustup

if [ $? -ne 0 ]; then
    echo "Error: install_rustup failed."
    echo "Abort"
    exit 1
fi
