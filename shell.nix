{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python310;
  pythonPackages = python.pkgs;
  llvm = if pkgs ? llvmPackages_20 then pkgs.llvmPackages_20 else pkgs.llvmPackages;
  runtimeLibs = with pkgs; [
    openssl
    libhwloc
    numactl
    tbb
    capstone
    llvm.libcxx
    llvm.libcxxabi
  ];
in
pkgs.mkShell {
  name = "tt-metal-dev-shell";

  packages = [
    pkgs.git
    pkgs.coreutils
    pkgs.gawk
    pkgs.gnused
    pkgs.gnugrep
    pkgs.findutils
    pkgs.diffutils
    pkgs.patch
    pkgs.gnutar
    pkgs.gzip
    pkgs.bzip2
    pkgs.file
    pkgs.perl
    pkgs.binutils
    pkgs.gnumake
    pkgs.cmake
    pkgs.ninja
    pkgs.pkg-config
    pkgs.pandoc
    pkgs.xz
    pkgs.jq
    pkgs.curl
    pkgs.wget
    pkgs.which
    pkgs.vim
    pkgs.uv
    python
    pythonPackages.pip
    pythonPackages.setuptools
    pythonPackages.wheel
    pythonPackages.virtualenv
    pythonPackages."setuptools-scm"
    pkgs.openssl
    pkgs.libhwloc
    pkgs.numactl
    pkgs.tbb
    pkgs.capstone
    llvm.clang
    llvm."clang-tools"
    llvm.lld
    llvm.libcxx
    llvm.libcxxabi
  ];

  shellHook = ''
    export TT_METAL_HOME="$PWD"
    export TT_METAL_RUNTIME_ROOT="$PWD"
    export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
    export CC="clang"
    export CXX="clang++"
    export VENV_PYTHON_VERSION="3.10"
    extra_lib_path="${pkgs.lib.makeLibraryPath runtimeLibs}"
    if [ -n "''${LD_LIBRARY_PATH:-}" ]; then
      export LD_LIBRARY_PATH="$extra_lib_path:$LD_LIBRARY_PATH"
    else
      export LD_LIBRARY_PATH="$extra_lib_path"
    fi

    echo "tt-metal Nix dev shell ready."
    echo "Repository root: $TT_METAL_HOME"
    echo "Suggested next step: ./create_venv.sh --python-version 3.10"
  '';
}
