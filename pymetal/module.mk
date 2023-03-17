include pymetal/csrc/module.mk

$(PYTHON_ENV)/lib/python3.8/site-packages/ttmetal.egg-link: python_env pymetal/csrc/setup_inplace_link
	bash -c "source $(PYTHON_ENV)/bin/activate; cd pymetal; pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu"

pymetal: pymetal/csrc $(PYTHON_ENV)/lib/python3.8/site-packages/ttmetal.egg-link;
