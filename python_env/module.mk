# Every variable in subdir must be prefixed with subdir (emulating a namespace)
PYTHON_ENV = $(OUT)/python_env

# Each module has a top level target as the entrypoint which must match the subdir name
python_env: $(PYTHON_ENV)/.installed

# .PRECIOUS: $(PYTHON_ENV)/.installed $(PYTHON_ENV)/%
$(PYTHON_ENV)/.installed: python_env/requirements.txt
	python3.8 -m venv $(PYTHON_ENV)
	bash -c "source $(PYTHON_ENV)/bin/activate && pip3.8 install wheel==0.37.1"
	bash -c "source $(PYTHON_ENV)/bin/activate && pip3.8 install -r python_env/requirements.txt"
	touch $@

$(PYTHON_ENV)/%: $(PYTHON_ENV)/.installed
