include libs/tt_lib/csrc/module.mk

libs/tt_lib: tt_lib/csrc

libs/tt_lib/dev_install: tt_lib/csrc/setup_local_so
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"
