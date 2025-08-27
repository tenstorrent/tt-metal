# yolov6l Web Demo

## How to run the web demo

### Server side:

- ssh into the server specifying the port:
  ```
  ssh -L 7000:localhost:7000 user@IP.ADDRESS
  ```

- After building metal, once you activate your python env. pip install the requirements on the server side:
  ```
  pip install -r models/demos/yolov6l/web_demo/server/requirements.txt
  ```

- From the server run:
  ```
  source models/demos/yolov6l/web_demo/server/run_uvicorn.sh
  ```

### Client side:

- git clone metal repo locally/on client side as well.
  ```
  cd models/demos/yolov6l/web_demo/client
  ```
- you may create a python virtual env and pip install the client side requirements.

  ```
  pip install -r requirements.txt
  ```
- on the client side run:
  ```
  source run_on_client_YOLOv6l_Metal --api-url http://IP.ADDRESS:7000
  ```
- a browser should automatically open and you will see the live object detection demo using your local camera.
