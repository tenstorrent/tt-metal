
# can maybe use find_package(Python3 REQUIRED COMPONENTS Interpreter) ?? not sure haven't looked into it enough

set(PYTHON_ENV_DIR ${CMAKE_BINARY_DIR}/python_env)

add_custom_target(python_env
    COMMAND python3 -m venv ${PYTHON_ENV_DIR}
    COMMAND bash -c "source ${PYTHON_ENV_DIR}/bin/activate && python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu"
    COMMENT "Creating Python virtual environment: ${PYTHON_ENV_DIR}"
    VERBATIM
)

add_custom_target(pip_deps
    COMMAND bash -c "source ${PYTHON_ENV_DIR}/bin/activate && python3 -m pip install -r ${CMAKE_SOURCE_DIR}/tt_metal/python_env/requirements-dev.txt"
    COMMENT "Installing Python requirements"
    DEPENDS python_env
    VERBATIM
)
