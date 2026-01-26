# Installation Guide for tt-train

This guide provides instructions for installing tt-train's ttml python module via pip.

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

Once tt-metal is installed and your virtual environment is activated, choose one of the following installation methods:

### Method A: pip install (standalone build)

**Regular installation:**

```bash
pip install /path/to/tt-train/
```

**Editable installation (for development):**

```bash
pip install --no-build-isolation -e /path/to/tt-train/
```

Use the editable installation (`-e`) if you plan to modify the tt-train source code and want changes to be reflected immediately without reinstalling.

> **Note:** This method runs its own CMake build. If you've already built tt-train via `build_metal.sh --build-tt-train`, use Method B instead to avoid rebuilding.

---

### Method B: Using pre-built ttml (recommended for development)

If you built tt-metal with tt-train support using `build_metal.sh --build-tt-train` or `--build-all`, ttml is already compiled. You can make Python find it by creating two `.pth` files:

```bash
# One-time setup - adds paths to your virtualenv's Python path

# 1. Add ttml Python source code
echo "/path/to/tt-metal/tt-train/sources/ttml" > python_env/lib/python3.10/site-packages/ttml.pth

# 2. Add the built _ttml C++ extension (.so file)
echo "/path/to/tt-metal/build/tt-train/sources/ttml" > python_env/lib/python3.10/site-packages/_ttml.pth
```

This approach:
- Avoids rebuilding when ttml is already built
- Reflects Python source changes immediately
- Works alongside the existing tt-metal editable install

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
