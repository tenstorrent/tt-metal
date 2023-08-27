include tt_eager/tt_lib/csrc/module.mk

tt_eager/tt_lib: tt_lib/csrc

tt_eager/tt_lib/dev_install: tt_lib/csrc/setup_local_so
	bash -c "source $(PYTHON_ENV)/bin/activate && pip install -e ."
