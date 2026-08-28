# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sentences per language, as data, tokenized at test time by the in-repo tokenizer.

`prompt_fixture.json` stores text WITH the token ids it produced and is the bit-exactness oracle;
this module is plain text, so breadth costs nothing.

Conventions: no digits in either form (the cleaner expands them and a recogniser writes them back as
numerals), ordinary prose, Arabic undiacritized, and sentences long enough to gate on.
"""

# Keyed by the language prefix of a voice name; "en" covers the unprefixed voices
# (neutral_*, casual_*, cheerful_*).
SENTENCES = {
    "en": [
        "It took me quite a long time to develop a voice, and now that I have it I am not going to be silent.",
        "The morning light came through the tall windows and settled on the wooden floor without a sound.",
        "She explained the whole arrangement twice, patiently, until everyone at the table understood it.",
    ],
    "fr": [
        "Il m'a fallu longtemps pour trouver ma voix, et maintenant que je l'ai, je ne me tairai pas.",
        "La lumière du matin traversait les hautes fenêtres et se posait doucement sur le plancher.",
    ],
    "de": [
        "Ich habe lange gebraucht, um meine Stimme zu finden, und jetzt werde ich nicht mehr schweigen.",
        "Das Morgenlicht fiel durch die hohen Fenster und legte sich lautlos auf den hölzernen Boden.",
    ],
    "es": [
        "Me llevó mucho tiempo encontrar mi voz, y ahora que la tengo no voy a quedarme callada.",
        "La luz de la mañana entraba por las ventanas altas y se posaba en el suelo de madera.",
    ],
    "it": [
        "Mi è servito molto tempo per trovare la mia voce, e ora che l'ho trovata non resterò in silenzio.",
        "La luce del mattino entrava dalle finestre alte e si posava senza rumore sul pavimento di legno.",
    ],
    "pt": [
        "Levei muito tempo para encontrar a minha voz, e agora que a tenho não vou ficar calada.",
        "A luz da manhã entrava pelas janelas altas e assentava sem ruído no chão de madeira.",
    ],
    "nl": [
        "Het heeft me lang gekost om mijn stem te vinden, en nu ik hem heb zal ik niet zwijgen.",
        "Het ochtendlicht viel door de hoge ramen en legde zich geluidloos op de houten vloer.",
    ],
    "hi": [
        "मुझे अपनी आवाज़ पाने में बहुत समय लगा, और अब जब मेरे पास है तो मैं चुप नहीं रहूँगी।",
        "सुबह की रोशनी ऊँची खिड़कियों से आकर लकड़ी के फ़र्श पर बिना आवाज़ ठहर गई।",
    ],
    "ar": [
        "استغرق الأمر وقتا طويلا حتى وجدت صوتي، والآن بعد أن وجدته لن أصمت.",
        "كان ضوء الصباح يعبر النوافذ العالية ويستقر على الأرضية الخشبية بلا صوت.",
    ],
}


def lang_of(voice):
    """Voice name -> corpus key. `ar_male` -> "ar"; `neutral_male` -> "en"."""
    head = voice.split("_")[0]
    return head if head in SENTENCES else "en"


def sentences_for(voice):
    return SENTENCES[lang_of(voice)]


def first_sentence_for(voice):
    return sentences_for(voice)[0]


# Longer sentences, five per language of growing length, for the WER gate. Kept apart from
# SENTENCES because that set is deliberately short -- these have to be long enough that one wrong
# word does not dominate the rate, and the gate is a per-language mean over speakers x sentences.
#
# Identical text to the sibling xtts_v2 port's `language_corpus.WER_SENTENCES`, deliberately: the
# same sentences through both models make the two WER numbers directly comparable.
WER_SENTENCES = {
    "en": [
        "The old map showed three islands that no sailor had ever found, and nobody wanted to be the "
        "first to erase them.",
        "Warm bread and strong coffee filled the small kitchen long before the sun came up, and the "
        "kettle went on again the moment the first pot stood empty.",
        "The archive filled four floors of a building designed for something else entirely, and every "
        "year the collection pressed a little harder against the walls until the shelves reached the "
        "ceiling.",
        "The bakery on the corner opens before dawn, and by six the whole street smells of warm bread, "
        "which is why the regulars arrive in the same order every morning without ever arranging it.",
        "Long before the railway reached the valley, the mail came over the mountain pass on horseback, "
        "and the timetable was a matter of weather rather than clocks, so the villagers learned to read "
        "the sky the way other people read a schedule.",
    ],
    "de": [
        "Der alte Stadtplan zeigte drei Inseln, die kein Seemann je gefunden hatte, und niemand wollte "
        "der erste sein, der sie wieder auslöschte.",
        "Warmes Brot und starker Kaffee füllten die kleine Küche lange bevor die Sonne aufging, und der "
        "Kessel kam wieder auf den Herd, sobald die erste Kanne leer stand.",
        "Das Archiv füllte vier Stockwerke eines Gebäudes, das für etwas völlig anderes entworfen worden "
        "war, und jedes Jahr drückte die Sammlung ein wenig härter gegen die Wände.",
        "Die Bäckerei an der Ecke öffnet vor dem Morgengrauen, und die ganze Straße riecht dann nach "
        "warmem Brot, weshalb die Stammgäste jeden Morgen in derselben Reihenfolge erscheinen.",
        "Lange bevor die Eisenbahn das Tal erreichte, kam die Post zu Pferd über den Bergpass, und der "
        "Fahrplan war eine Frage des Wetters und nicht der Uhren, sodass die Dorfbewohner lernten, den "
        "Himmel zu lesen.",
    ],
    "fr": [
        "La vieille carte montrait trois îles qu'aucun marin n'avait jamais trouvées, et personne ne "
        "voulait être le premier à les effacer.",
        "Le pain chaud et le café fort remplissaient la petite cuisine bien avant le lever du soleil, et "
        "la bouilloire repartait dès que la première cafetière restait vide.",
        "Les archives occupaient quatre étages d'un bâtiment conçu pour tout autre chose, et chaque année "
        "la collection pressait un peu plus fort contre les murs.",
        "La boulangerie du coin ouvre avant l'aube, et toute la rue sent alors le pain chaud, ce qui "
        "explique pourquoi les habitués arrivent dans le même ordre chaque matin.",
        "Bien avant que le chemin de fer n'atteigne la vallée, le courrier passait le col à cheval, et "
        "l'horaire dépendait du temps plutôt que des horloges, si bien que les villageois apprirent à "
        "lire le ciel.",
    ],
    "es": [
        "El mapa antiguo mostraba tres islas que ningún marinero había encontrado nunca, y nadie quería "
        "ser el primero en borrarlas.",
        "El pan caliente y el café fuerte llenaban la pequeña cocina mucho antes de que saliera el sol, y "
        "la tetera volvía al fuego en cuanto la primera jarra quedaba vacía.",
        "El archivo ocupaba cuatro plantas de un edificio diseñado para algo completamente distinto, y "
        "cada año la colección apretaba un poco más contra las paredes.",
        "La panadería de la esquina abre antes del amanecer, y toda la calle huele entonces a pan "
        "caliente, por lo que los clientes habituales llegan en el mismo orden cada mañana.",
        "Mucho antes de que el ferrocarril llegara al valle, el correo cruzaba el puerto de montaña a "
        "caballo, y el horario era una cuestión del tiempo y no de los relojes, así que los aldeanos "
        "aprendieron a leer el cielo.",
    ],
    "it": [
        "La vecchia mappa mostrava tre isole che nessun marinaio aveva mai trovato, e nessuno voleva "
        "essere il primo a cancellarle.",
        "Il pane caldo e il caffè forte riempivano la piccola cucina molto prima che il sole si alzasse, "
        "e il bollitore tornava sul fuoco appena la prima caffettiera restava vuota.",
        "L'archivio occupava quattro piani di un edificio progettato per qualcosa di completamente "
        "diverso, e ogni anno la collezione premeva un poco più forte contro le pareti.",
        "La panetteria all'angolo apre prima dell'alba, e tutta la strada profuma allora di pane caldo, "
        "ed è per questo che i clienti abituali arrivano nello stesso ordine ogni mattina.",
        "Molto prima che la ferrovia raggiungesse la valle, la posta passava il colle a cavallo, e "
        "l'orario era una questione di tempo e non di orologi, così gli abitanti impararono a leggere il "
        "cielo.",
    ],
    "pt": [
        "O mapa antigo mostrava três ilhas que nenhum marinheiro jamais encontrara, e ninguém queria ser "
        "o primeiro a apagá-las.",
        "O pão quente e o café forte enchiam a pequena cozinha muito antes de o sol nascer, e a chaleira "
        "voltava ao fogo assim que o primeiro bule ficava vazio.",
        "O arquivo ocupava quatro andares de um edifício projetado para algo completamente diferente, e "
        "cada ano a coleção pressionava um pouco mais contra as paredes.",
        "A padaria da esquina abre antes do amanhecer, e toda a rua fica então com cheiro de pão quente, "
        "e é por isso que os fregueses chegam na mesma ordem todas as manhãs.",
        "Muito antes de a ferrovia chegar ao vale, o correio atravessava a serra a cavalo, e o horário "
        "era uma questão de tempo e não de relógios, de modo que os moradores aprenderam a ler o céu.",
    ],
    "nl": [
        "De oude kaart toonde drie eilanden die geen enkele zeeman ooit had gevonden, en niemand wilde de "
        "eerste zijn die ze uitwiste.",
        "Warm brood en sterke koffie vulden de kleine keuken lang voordat de zon opkwam, en de ketel ging "
        "weer op het vuur zodra de eerste kan leeg stond.",
        "Het archief vulde vier verdiepingen van een gebouw dat voor iets heel anders was ontworpen, en "
        "elk jaar drukte de collectie een beetje harder tegen de muren.",
        "De bakkerij op de hoek opent voor zonsopgang, en de hele straat ruikt dan naar warm brood, en "
        "daarom komen de vaste klanten elke ochtend in dezelfde volgorde.",
        "Lang voordat de spoorlijn de vallei bereikte, kwam de post te paard over de bergpas, en de "
        "tijdtabel was een kwestie van het weer en niet van klokken, zodat de dorpelingen leerden de "
        "lucht te lezen.",
    ],
    "hi": [
        "पुराने नक्शे में तीन ऐसे द्वीप दिखते थे जिन्हें कोई नाविक कभी नहीं खोज पाया, और कोई भी उन्हें "
        "सबसे पहले मिटाने वाला नहीं बनना चाहता था।",
        "गरम रोटी और तेज़ चाय सूरज निकलने से बहुत पहले ही छोटी रसोई को भर देती थी, और पहला बरतन खाली होते "
        "ही केतली फिर आँच पर चढ़ जाती थी।",
        "वह संग्रह ऐसी इमारत की चार मंज़िलों में फैला था जो किसी और काम के लिए बनी थी, और हर साल संग्रह "
        "दीवारों पर थोड़ा और ज़ोर डालता जाता था।",
        "नुक्कड़ की दुकान भोर से पहले खुल जाती है, और पूरी गली में गरम रोटी की महक फैल जाती है, इसलिए "
        "पुराने ग्राहक हर सुबह उसी क्रम में आते हैं।",
        "रेल के घाटी तक पहुँचने से बहुत पहले, डाक घोड़े पर पहाड़ी दर्रे से आती थी, और समय सारणी घड़ियों "
        "से कम और मौसम से ज़्यादा तय होती थी, इसलिए गाँव वालों ने आसमान पढ़ना सीख लिया।",
    ],
    "ar": [
        "أظهرت الخريطة القديمة ثلاث جزر لم يجدها أي بحار من قبل، ولم يرغب أحد في أن يكون أول من يمحوها.",
        "كان الخبز الساخن والقهوة القوية يملأان المطبخ الصغير قبل شروق الشمس بوقت طويل، وكانت الغلاية "
        "تعود إلى النار بمجرد أن يفرغ الإبريق الأول.",
        "شغل الأرشيف أربعة طوابق من مبنى صمم لشيء آخر تماما، وكانت المجموعة تضغط على الجدران بقوة أكبر "
        "قليلا كل عام.",
        "يفتح المخبز في الزاوية قبل الفجر، ويفوح الشارع كله برائحة الخبز الساخن، ولهذا يأتي الزبائن "
        "المعتادون في الترتيب نفسه كل صباح.",
        "قبل أن يصل القطار إلى الوادي بوقت طويل، كان البريد يعبر الممر الجبلي على ظهر حصان، وكان الجدول "
        "يعتمد على الطقس لا على الساعات، فتعلم القرويون قراءة السماء.",
    ],
}


def wer_sentences_for(voice):
    """-> the five WER sentences for this voice's language."""
    return WER_SENTENCES[lang_of(voice)]
