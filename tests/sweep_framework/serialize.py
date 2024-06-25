from ttnn import *
import tt_lib


def serialize(object):
    if "attribute_names" in dir(object) and "attribute_values" in dir(object):
        serialized_object = dict()
        serialized_object["type"] = str(type(object)).split("'")[1]
        attr_names = object.attribute_names
        attr_values = object.attribute_values()
        for i in range(len(attr_names)):
            serialized_object[attr_names[i]] = serialize(attr_values[i])
        return serialized_object
    else:
        return str(object)


def deserialize(object):
    if isinstance(object, dict):
        type = eval(object["type"])
        object.pop("type")
        deserialized_parameters = dict()
        for elem in object:
            deserialized_parameters[elem] = deserialize(object[elem])
        return type(**deserialized_parameters)
    else:
        try:
            return eval(object)
        except:
            return str(object)
