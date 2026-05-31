{ lib, python3, buildPythonPackage, tt-metal, numpy, loguru, networkx
, graphviz, pyyaml, click, pandas, seaborn, ml_dtypes
}:

buildPythonPackage rec {
  pname = "ttnn";
  version = "0.58.0";

  src = builtins.path { path = ./..; name = "ttnn-source"; };

  pyproject = true;

  build-system = [
    python3.pkgs.setuptools
    python3.pkgs.wheel
    python3.pkgs.setuptools-scm
  ];

  dependencies = [
    numpy
    loguru
    networkx
    graphviz
    pyyaml
    click
    pandas
    seaborn
    ml_dtypes
  ];

  # The actual C++ shared lib is provided by tt-metal
  TTMETAL_LIB_DIR = "${tt-metal}/lib";

  preBuild = ''
    export CPATH="${tt-metal}/include:$CPATH"
    export LIBRARY_PATH="${tt-metal}/lib:$LIBRARY_PATH"
  '';

  meta = with lib; {
    description = "TTNN: Python bindings for TT-Metalium";
    homepage = "https://github.com/tenstorrent/tt-metal";
    license = licenses.asl20;
    maintainers = with maintainers; [ ];
    platforms = [ "x86_64-linux" ];
  };
}
