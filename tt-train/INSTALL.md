# Installation Guide for tt-train

This guide provides instructions for installing tt-train via pip.

## Prerequisites

### System Requirements
- Python 3.10 or higher
- Ubuntu 20.04+ (or compatible Linux distribution)
- Tenstorrent hardware (for running on device)

## Step 1: Install tt-metal (Prerequisite)

Before installing tt-train, you must have tt-metal installed. Choose one of the following options:

### Option A: Install tt-metal via pip (Recommended)

Use the tt-metal venv creation script to set up your Python environment:

```bash
cd /path/to/tt-metal
./create_venv.sh
```

This script will:
- Create a virtual environment in `python_env/`
- Install the correct pip, setuptools, and wheel versions for your OS
- Install development dependencies
- Install tt-metal (ttnn) in editable mode

Activate the environment:

```bash
source python_env/bin/activate
```

Your terminal prompt should now show `(python_env)` indicating the virtual environment is active.

---

### Option B: Install tt-metal via APT

Install tt-metal from the APT repository:

```bash
# Add the Tenstorrent APT repository (if not already added)
# Instructions for adding the repository should be provided by Tenstorrent

# Install tt-metal (ttnn)
sudo apt update
sudo apt install ttnn
```

Create and activate a Python virtual environment:

```bash
python3 -m venv python_env
source python_env/bin/activate
```

---

### Option C: Build tt-metal from Source

Clone and build tt-metal:

```bash
cd /path/to/tt-metal

# Initialize submodules
git submodule update --init --recursive

# Build tt-metal
./build_metal.sh
```

For additional build options, see the tt-metal documentation.

Set up the Python virtual environment using the tt-metal venv creation script:

```bash
cd /path/to/tt-metal
./create_venv.sh
```

This will create the virtual environment and install tt-metal in editable mode.

Activate the environment:

```bash
source python_env/bin/activate
```

---

## Step 2: Install tt-train

Once tt-metal is installed and your virtual environment is activated, install tt-train:

**Regular installation:**

```bash
pip install /path/to/tt-metal/tt-train/
```

**Editable installation (for development):**

```bash
pip install -e /path/to/tt-metal/tt-train/
```

Use the editable installation (`-e`) if you plan to modify the tt-train source code and want changes to be reflected immediately without reinstalling.

---

## Step 3: Set Environment Variables

Set the required environment variables:

```bash
export TT_METAL_HOME=/path/to/your/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

You may want to add these to your `~/.bashrc` or `~/.bash_profile` to make them persistent across sessions.

---

## Step 4: Verify Installation

Verify the installation by importing the modules in Python:

```python
import ttnn
print("successfully imported ttnn")

import ttml
print("successfully imported ttml")
```

To check installed locations:

```python
import ttnn
print(f"ttnn location: {ttnn.__file__}")

import ttml
print(f"ttml location: {ttml.__file__}")
```
