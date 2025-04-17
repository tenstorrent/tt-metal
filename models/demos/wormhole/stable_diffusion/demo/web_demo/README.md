# Stable Diffusion Web Demo

## Introduction
Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. These instructions pertain to running Stable Diffusion with an interactive web interface.

## How to Run

> [!NOTE]
>
> If you are using Wormhole, you must set the `WH_ARCH_YAML` environment variable.
>
> ```
> export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
> ```

> In order to post the generated image from the SD model to your local computer, make sure to ssh into the port 8501:
> ```
> ssh -L8501:localhost:8501 username@IP_ADDRESS
> ```


To run the demo, make sure to build the project, activate the environment, and set the appropriate environment variables.
For more information, refer [installation and build guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).

The interface of this project is built with [Streamlit](https://streamlit.io). The web server utilizes [Flask](https://flask.palletsprojects.com/en/3.0.x/) and a [Gunicorn](https://gunicorn.org) server to server a WSGI application.

### Install dependencies
Use `pip install -r models/demos/wormhole/stable_diffusion/demo/web_demo/requirements.txt` to install the dependencies on your machine.

### Invoke web demo
Use `python models/demos/wormhole/stable_diffusion/demo/web_demo/web_demo.py` to run the web demo. It should automatically pop-up at this [address](http://localhost:8501)(localhost:8501).
