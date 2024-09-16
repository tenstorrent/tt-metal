# Python Packaging and Wheels

We will be going over:

- some brief context over Python packaging and the wheel format
- how we structure our wheel build and packaging for TT-NN
- some of our pitfalls and potential solutions / long-term fixes

## Context

### What is the purpose of a wheel?

A wheel is really just a zip file containing all Python files and binaries needed to install a Python package. You could think of the wheel as the binary format of the Python world.

### What are some important concepts of the wheel?

- Wheels should contain all files and binaries needed for a Python package installation. This is because all that really happens at install time is an unzip and copy.
- Projects should be structured so that the wheel build is hermetic. One way the Python ecosystem encourages this is by encouraging the use of build tools that will take a source distribution (srcdir) of a (similar to but different from the source repository of a project) and build a wheel directly from it in a hermetic virtual environment.
- Wheel builds should be as declarative as possible. This is obviously more difficult as your build gets more complicated. A common and definitely relatable example is a Python project which has C++ extensions.

### What are you trying to solve specifically for TT-NN with a wheel?

There are various things we want to solve, such as:

- The most important thing: enabling users to install our Python package without having to build from source.
- A second important, related thing to the first: conforming to popular build patterns for Python projects with complex C++ extensions and establishing patterns for patterns which haven't been fully agreed upon yet.
- Enabling upload to Python package registries, a big example being PyPI.
- Ensuring our Python build conforms as much as possible to a reproducible Python build that the wider Python community can understand, and more importantly, contribute to.

### What will this document cover?

We will be going over:
- How we ensure users can conveniently use our TT-NN Python API and underlying C++ bindings if they install from a wheel.
- How we ensure that users' setups mirror an internal developer's setup as closely as possible so that we catch errors related to both functionality and packaging infrastructure early.
- How we deal with build and runtime quirks that don't translate well to a potential user's production environment.

### What will this document NOT cover?

- We will not going over the PEPs, including those that relate to what we're talking about here. We may reference certain concepts in the PEPs to help explain and provide context about the recommendations and rules around certain ideas or procedures.

## The details

There are three very key source files that will attract virtually all our attention here:

- `setup.py`
- `ttnn/ttnn/library_tweaks.py`
- `pyproject.toml`

These two files do the bulk of the heavy lifting when it comes to ensuring that we have a working wheel.

However, instead of going through these files procedurally, we will instead go over major concepts and fixes and explain how these files (and potentially other minor ones) play into our needs.

### A little background on Python packages

When you run Python and use one of its various packages, especially those you install from the community, the local Python first needs to find and load those requested packages. For example, when you do:

```python
import ttnn
```

Where does Python look for `ttnn`? In a typical Python installation on a Linux system, the path at which `ttnn` may be stored could look like:

```
/usr/lib/python3.8/site-packages/ttnn/
```

This means all the Python files and C++ bindings for the library would be stored in that directory.

This also implies that a path like `/usr/lib/python3.8/site-packages/` would be included in **search directory** for Python packages. Note that the path could look slightly different for a virtual environment (by `venv`) Python installation, for a Conda installation etc.

What this ultimately means for us is that we need to ensure that our installation works no matter the ultimate package resting spot for our installation files.

### The development environment

However, most internal TT-NN developers do not install our package into their Python environment's packages directory. If not, then how exactly do TT-NN developers interact with their own package? The answer is the **editable install**.

Python's editable install is a feature that enables developer's to rapidly make changes to Python files and see the results immediately. You can do an editable install with the `-e` / `--editable` flag at `pip install` time with the source repo. Note that this is also how we install TT-NN into the developer virtual environment (ie. `create_venv.sh`). Without the virtual install, developers would have to go through the `pip` install process every change.

The problem this poses is that the environment in the source repo can potentially encourage developers to create behaviour in TT-NN that may not translate very well to a typical production Python environment. Specifically, because developers are so used to a sandbox environment where they have access to all the tools and **directory structure** of the original repository, it becomes easier to assume users may have certain capabilities that they don't actually have. In an upcoming section, we'll go over a key example that highlights this problem, and some things we did to get around it.

### The `TT_METAL_HOME` directory

## Lingering issues

UNDER CONSTRUCTION

### Cross-platform and Cross-OS builds

Ubuntu 20.04/22.04/24.04 has been a bit of a headache

### Runtime system dependencies

ensuring users have compatible versions of of c++ stdlib, other c++-level dependencies

### Lifting up more build-time dependencies

from underlying host to declaratively keeping them in source