# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYTHON_ENV = $(OUT)/python_env

# Each module has a top level target as the entrypoint which must match the subdir name
python_env: $(PYTHON_ENV)/.installed

python_env/dev: $(PYTHON_ENV)/.installed-dev

# .PRECIOUS: $(PYTHON_ENV)/.installed $(PYTHON_ENV)/%
$(PYTHON_ENV)/.installed:
	python3.8 -m venv $(PYTHON_ENV)
	bash -c "source $(PYTHON_ENV)/bin/activate && python -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu"
	touch $@

$(PYTHON_ENV)/%: $(PYTHON_ENV)/.installed
	bash -c "source $(PYTHON_ENV)/bin/activate"

$(PYTHON_ENV)/.installed-dev: libs/tt_lib/dev_install python_env tt_metal/python_env/requirements-dev.txt
	bash -c "source $(PYTHON_ENV)/bin/activate && python -m pip install -r tt_metal/python_env/requirements-dev.txt"
	touch $@
