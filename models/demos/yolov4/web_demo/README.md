# Yolov4 Web Demo

## How to run the web demo

### Server side:

- ssh into the server specifying the port:
  ```
  ssh -L 7000:localhost:7000 user@IP.ADDRESS
  ```

- After building metal, once you activate your python env. pip install the requirements on the server side:
  ```
  pip install -r models/demos/yolov4/web_demo/server/requirements.txt
  ```

- After installing the server side requirments, ONLY if you are running the demo on an N300 card,run the following to export the approprite envirement variable for N300.
  ```
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
  ```

- From the server run:
  ```
  source models/demos/yolov4/web_demo/server/run_uvicorn.sh
  ```

### Client side:

- git clone metal repo locally/on client side as well.
  ```
  cd models/demos/yolov4/web_demo/client
  ```
- you may create a python virtual env and pip install the client side requirements.

  ```
  pip install -r models/demos/yolov4/web_demo/client/requirements.txt
  ```
- on the client side run:
  ```
  source run_on_client_YOLOv4_Metal --api-url http://IP.ADDRESS:7000
  ```
- a browser should automatically open and you will see the live object detection demo using your local camera.
