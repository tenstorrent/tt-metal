{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "tt-metal-dev";

  buildInputs = with pkgs; [
    # Build tools
    cmake
    gcc
    gnumake
    pkg-config

    # Python environment
    python3
    python3Packages.pip
    python3Packages.virtualenv

    # Required libraries
    libffi
    openssl
    zlib

    # Development tools
    git
    gdb
    valgrind

    # Documentation
    doxygen
    graphviz
  ];

  shellHook = ''
    echo "🖥️  tt-metal development environment"
    echo "   Python: $(python3 --version)"
    echo "   GCC:    $(gcc --version | head -1)"

    # Set up Python virtual environment
    if [ ! -d .venv ]; then
      python3 -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip
    else
      source .venv/bin/activate
    fi

    # NixOS: FHS workaround
    if [ -f /etc/NIXOS ]; then
      export TT_HOME=$(pwd)
      echo "   NixOS detected — TT_HOME=$TT_HOME"
    fi
  '';
}
