# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYTHON_ENV ?= $(OUT)/python_env

# Each module has a top level target as the entrypoint which must match the subdir name
python_env: $(PYTHON_ENV)/.installed

python_env/dev: $(PYTHON_ENV)/.installed-dev

python_env/clean:
	rm -rf $(PYTHON_ENV)

# .PRECIOUS: $(PYTHON_ENV)/.installed $(PYTHON_ENV)/%
$(PYTHON_ENV)/.installed:
	python3 -m venv $(PYTHON_ENV)
	bash -c "source $(PYTHON_ENV)/bin/activate && python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu"
	echo "Installing python env build backend requirements..."
	bash -c "source $(PYTHON_ENV)/bin/activate && python3 -m pip install setuptools wheel"
	touch $@

$(PYTHON_ENV)/%: $(PYTHON_ENV)/.installed
	bash -c "source $(PYTHON_ENV)/bin/activate"

ifdef TT_METAL_ENV_IS_DEV
# Once we split this out and put this python_env module.mk declaration at the end, then we'll actually properly depend on these local sos being installed
$(PYTHON_ENV)/.installed-dev: $(PYTHON_ENV)/.installed $(TT_LIB_LIB_LOCAL_SO) $(TTNN_PYBIND11_LOCAL_SO) tt_metal/python_env/requirements-dev.txt
else
$(PYTHON_ENV)/.installed-dev: $(PYTHON_ENV)/.installed tt_metal/python_env/requirements-dev.txt
endif
	echo "Installing dev environment packages..."
	bash -c "source $(PYTHON_ENV)/bin/activate && python3 -m pip install -r tt_metal/python_env/requirements-dev.txt"
	echo "Installing editable dev version of tt_eager packages..."
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e ."
	echo "Installing editable dev version of ttnn package..."
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e ttnn"
	touch $@
	echo "Generating stubs..."
	bash -c "source $(PYTHON_ENV)/bin/activate && stubgen -m tt_lib -m tt_lib.device -m tt_lib.profiler -m tt_lib.tensor -m tt_lib.dtx -m tt_lib.operations -m tt_lib.operations.primary -m tt_lib.operations.primary.transformers -o tt_eager"
	bash -c "source $(PYTHON_ENV)/bin/activate && stubgen -p ttnn._ttnn -o ttnn"
	bash -c "sed -i 's/\._C/tt_lib/g' tt_eager/tt_lib/__init__.pyi"
